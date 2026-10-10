"""One-factor-at-a-time experiments on single (wall, source) variants, with on-disk caching and metrics.

Typical use (see notebooks/01_factor_influence.ipynb):

    meta = load_meta(overrides=[...])
    base = sample_draw(meta, seed)
    ref = run_variant(base, meta, wall="bare", source="impact", freq=1000)
    var = run_variant(vary(base, {"tunnel.curvature_horizontal": 1 / 300}), meta, wall="bare", source="impact", freq=1000)
    rows = compare(ref, var)
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .build import build_model, make_domain, paint_rock, tunnel_geometry_fields
from .simulate import SOURCE_KIND, eikonal, frequency_cap, impact_directions, nudge_into_rock, positions, propagate

COMP_NAMES = ("p", "s", "n", "v")
CODE_VERSION = 2          # bump when simulation code changes, invalidates the cache


# --------------------------------------------------------------------------- #
def vary(params: dict, changes: dict) -> dict:
    """Deep copy of a draw with dotted-path values replaced, e.g. {"tunnel.roughness.amplitude": 0.0}.
    A callable value is applied to the dict at that path: {"rock.structures": lambda lst: lst + [...]}"""
    p = copy.deepcopy(params)
    for path, val in changes.items():
        d = p
        keys = path.split(".")
        for k in keys[:-1]:
            d = d[int(k)] if isinstance(d, list) else d[k]
        last = int(keys[-1]) if isinstance(d, list) else keys[-1]
        d[last] = val(d[last]) if callable(val) else val
    return p


def vary_meta(meta: dict, changes: dict) -> dict:
    return vary(meta, changes)


def _key(*objs) -> str:
    return hashlib.sha1(json.dumps(objs, sort_keys=True, default=float).encode()).hexdigest()[:16]


def run_variant(params: dict, meta: dict, dims: int = 2, wall: str = "bare", source: str = "impact",
                freq: float | None = None, device: str = "cpu", chunk: int | None = None,
                cache_dir: str | Path = "runs/experiments", verbose: bool = True, label: str = "") -> dict:
    """Simulate one (wall, source) variant of a draw. Results are cached by a hash of all inputs."""
    sim_meta = {k: meta[k] for k in ("grid", "record", "domain", "rock")}
    key = _key(params, sim_meta, dims, wall, source, freq, CODE_VERSION)
    path = Path(cache_dir) / f"{key}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        out = {k: z[k] for k in z.files}
        out["label"] = label
        if verbose:
            print(f"  [cached] {label or key}")
        return out

    t0 = time.perf_counter()
    dom, geom = make_domain(params, meta, dims)
    rock = paint_rock(params, meta, dom, geom)
    tf = tunnel_geometry_fields(params, dom, geom)
    model = build_model(params, meta, dims, wall, dom, geom, rock, tf)
    rcv, shots, _, _ = positions(params, geom)
    acq = params["acquisition"]
    blocked = model["excavated"] | (model["labels"] == 6)
    nudge_into_rock(rcv, blocked, dom, geom, [p["height"] for p in acq["receivers"] + acq["virtual"]])
    nudge_into_rock(shots, blocked, dom, geom, [p["height"] for p in acq["shots"]])
    dirs = impact_directions(params, geom, dims)
    cap = frequency_cap({wall: model}, meta, dom.dx)
    f = float(freq) if freq else min(params["sources"][source]["peak_freq"], cap)
    chunk = chunk or (len(shots) if dims == 2 else 2)
    data, t, wav, sps = propagate(model, dom, shots, rcv, SOURCE_KIND[source], f,
                                  params["sources"][source]["amplitude"], dirs, meta, device, chunk, verbose=False)
    eik = eikonal(model, dom, shots, rcv)
    out = {"data": data, "t": t, "eik": eik, "rcv": rcv, "shots": shots,
           "offset": np.linalg.norm(shots[:, None, :] - rcv[None, :, :], axis=-1),
           "freq": np.float64(f), "cap": np.float64(cap), "dx": np.float64(dom.dx), "dims": np.int64(dims),
           "seconds_per_shot": np.float64(sps), "runtime_s": np.float64(time.perf_counter() - t0),
           "cells": np.int64(np.prod(dom.shape)), "origin": dom.origin,
           "vp_plane": (model["vp"] if dims == 2 else model["vp"][:, :, int(round(-dom.origin[2] / dom.dx))]
                        ).astype(np.float16)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    out["label"] = label
    if verbose:
        print(f"  [ran {out['runtime_s']:.0f} s, {sps:.1f} s/shot, {out['cells'] / 1e6:.2f} M cells, "
              f"{f:.0f} Hz] {label or key}")
    return out


# --------------------------------------------------------------------------- #
def _moving_rms(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(n) / n
    return np.sqrt(np.convolve(x * x, k, mode="same"))


def fa_window(res: dict, s: int, r: int, pre_periods: float = 1.2, post_periods: float = 1.5):
    """Time window around the first arrival of the reference: [eik - 1.2 T, eik + 1.5 T], T = 1 / f."""
    T = 1.0 / float(res["freq"])
    e = res["eik"][s, r]
    return (res["t"] >= e - pre_periods * T) & (res["t"] <= e + post_periods * T)


def _xcorr_lag(a: np.ndarray, b: np.ndarray, dt: float, max_lag: int) -> float:
    """Lag (s) by which b is delayed relative to a, sub-sample by parabolic interpolation."""
    a = a - a.mean()
    b = b - b.mean()
    full = np.correlate(b, a, mode="full")
    mid = len(a) - 1
    lo, hi = mid - max_lag, mid + max_lag + 1
    c = full[lo:hi]
    k = int(np.argmax(c))
    frac = 0.0
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        frac = 0.5 * (c[k - 1] - c[k + 1]) / den if den != 0 else 0.0
    return (k - max_lag + frac) * dt


def nrms(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised RMS difference in percent: 0 identical, ~141 uncorrelated, 200 opposite polarity."""
    ra, rb = np.sqrt(np.mean(a * a)), np.sqrt(np.mean(b * b))
    return float(200.0 * np.sqrt(np.mean((a - b) ** 2)) / (ra + rb + 1e-30))


def compare(ref: dict, var: dict, comps=("s", "n")) -> list[dict]:
    """Per-trace metrics of `var` against `ref` (same receivers / shots, same time axis).

    dt_eik  : change of the eikonal traveltime [ms]
    dt_fa   : waveform delay in the first-arrival window, cross-correlation [ms] (+ = later)
    nrms_fa : NRMS difference in the first-arrival window [%]
    nrms    : NRMS difference over the whole trace [%]
    amp_fa  : first-arrival window RMS ratio var / ref [dB]
    """
    dt = float(ref["t"][1] - ref["t"][0])
    nt = min(ref["data"].shape[-1], var["data"].shape[-1])
    rows = []
    n_s, n_r = ref["data"].shape[:2]
    for c in comps:
        ci = COMP_NAMES.index(c)
        for s in range(n_s):
            for r in range(n_r):
                a, b = ref["data"][s, r, ci, :nt].astype(float), var["data"][s, r, ci, :nt].astype(float)
                w = fa_window(ref, s, r)[:nt]
                max_lag = int(round(0.5 / float(ref["freq"]) / dt)) + 2
                rows.append({
                    "comp": c, "shot": s, "rcv": r, "offset": float(ref["offset"][s, r]),
                    "dt_eik": 1e3 * float(var["eik"][s, r] - ref["eik"][s, r]),
                    "dt_fa": 1e3 * _xcorr_lag(a[w], b[w], dt, max_lag),
                    "nrms_fa": nrms(a[w], b[w]), "nrms": nrms(a, b),
                    "amp_fa": 20 * np.log10(np.sqrt(np.mean(b[w] ** 2)) / (np.sqrt(np.mean(a[w] ** 2)) + 1e-30) + 1e-30),
                })
    return rows


def fa_visibility(res: dict, comps=("s", "n", "p")) -> dict:
    """Median over traces of: strongest first-arrival-window level relative to the strongest level of
    the trace, in dB (moving RMS over one period). 0 dB = the first arrival is the strongest phase."""
    dt = float(res["t"][1] - res["t"][0])
    n = max(int(round(1.0 / float(res["freq"]) / dt)), 1)
    out = {}
    for c in comps:
        ci = COMP_NAMES.index(c)
        if ci >= res["data"].shape[2]:
            continue
        v = []
        for s in range(res["data"].shape[0]):
            for r in range(res["data"].shape[1]):
                m = _moving_rms(res["data"][s, r, ci].astype(float), n)
                w = fa_window(res, s, r)
                v.append(20 * np.log10(m[w].max() / (m.max() + 1e-30) + 1e-30))
        out[c] = float(np.median(v))
    return out


def summarise(rows: list[dict]) -> dict:
    """Medians of absolute timing changes and of the difference measures."""
    def med(k, absval=False):
        x = np.array([r[k] for r in rows], float)
        return float(np.median(np.abs(x) if absval else x))
    p90 = float(np.percentile(np.abs([r["dt_eik"] for r in rows]), 90))
    return {"|dt_eik| ms": med("dt_eik", True), "p90 |dt_eik| ms": p90, "|dt_fa| ms": med("dt_fa", True),
            "nrms_fa %": med("nrms_fa"), "nrms %": med("nrms"), "amp_fa dB": med("amp_fa")}


def table(summaries: dict, fmt: str = "{:.2f}") -> str:
    """Markdown table from {name: summary dict}."""
    names = list(summaries)
    cols = list(next(iter(summaries.values())))
    lines = ["| experiment | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(fmt.format(summaries[n][c]) for c in cols) + " |")
    return "\n".join(lines)
