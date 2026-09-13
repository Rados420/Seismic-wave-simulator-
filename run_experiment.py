#!/usr/bin/env python
"""Run one simulator experiment: paint -> propagate -> eikonal -> onset -> HDF5 + figures.

    python run_experiment.py --config configs/homogeneous.yaml [--out runs] [--shots N] [--dx 0.1] [--no-tunnel]
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import deepwave

from fasim.config import load_config, save_config
from fasim.geometry import Grid, receiver_names, receiver_positions, shot_positions, offsets
from fasim.painter import paint, min_wavelength
from fasim.propagate import run_shots, snapshots, COMPONENTS
from fasim.eikonal import eikonal_labels
from fasim.onset import pick_gather, wavelet_onset, emergence_delay
from fasim.io import save_run
from fasim.postprocess import apply as post_apply
from fasim import plot as fplot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None, help="run name (default: config name)")
    ap.add_argument("--shots", type=int, default=None, help="only simulate the first N shots")
    ap.add_argument("--dx", type=float, default=None, help="override grid spacing")
    ap.add_argument("--freq", type=float, default=None, help="override source peak frequency")
    ap.add_argument("--source", choices=["force", "pressure"], default=None)
    ap.add_argument("--no-tunnel", action="store_true", help="disable the cavity (sanity runs)")
    ap.add_argument("--picker", choices=["envelope_threshold", "amplitude_threshold", "sta_lta", "aic"], default=None)
    ap.add_argument("--threshold-ref", choices=["trace", "gather", "window"], default=None)
    ap.add_argument("--rel-threshold", type=float, default=None)
    ap.add_argument("--component", choices=["y", "x", "p", "vec"], default=None, help="primary onset component")
    ap.add_argument("--record-ms", type=float, default=None)
    ap.add_argument("--pml", type=int, default=None, help="override domain.pml_width [cells]")
    ap.add_argument("--q", type=float, default=None, help="constant-Q attenuation (post-processing)")
    ap.add_argument("--spreading", type=float, default=None, help="extra spreading exponent p: (t/1ms)^-p")
    ap.add_argument("--texture", type=float, default=None, help="override rock.texture_std")
    ap.add_argument("--texture-len", type=float, default=None, help="override rock.texture_corr_len [m]")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.shots:
        cfg.acquisition.n_shots = args.shots
        if cfg.acquisition.shot_z is not None:
            cfg.acquisition.shot_z = cfg.acquisition.shot_z[: args.shots]
    if args.dx:
        cfg.domain.dx = args.dx
    if args.freq:
        cfg.source.peak_freq = args.freq
    if args.source:
        cfg.source.kind = args.source
    if args.no_tunnel:
        cfg.tunnel.enabled = False
    if args.picker:
        cfg.labels.picker = args.picker
    if args.threshold_ref:
        cfg.labels.threshold_ref = args.threshold_ref
    if args.rel_threshold is not None:
        cfg.labels.rel_threshold = args.rel_threshold
    if args.component:
        cfg.labels.component = args.component
    if args.record_ms:
        cfg.sim.record_ms = args.record_ms
    if args.pml is not None:
        cfg.domain.pml_width = args.pml
    if args.q is not None:
        cfg.post.q = args.q if args.q > 0 else None
    if args.spreading is not None:
        cfg.post.spreading_exponent = args.spreading
    if args.texture is not None:
        cfg.rock.texture_std = args.texture
    if args.texture_len is not None:
        cfg.rock.texture_corr_len = args.texture_len
    name = args.name or cfg.name
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    grid = Grid.from_domain(cfg.domain)
    model = paint(cfg, grid, rng)
    src_pos = shot_positions(cfg.acquisition, cfg.tunnel)
    rcv_pos = receiver_positions(cfg.acquisition, cfg.tunnel)
    names = receiver_names(cfg.acquisition)
    lam_min = min_wavelength(model, 2.5 * cfg.source.peak_freq)
    print(f"[{name}] grid {grid.nz}x{grid.nx} @ {grid.dx} m, {src_pos.shape[0]} shots, {rcv_pos.shape[0]} rcv, "
          f"family={cfg.structure.family}, tunnel={cfg.tunnel.enabled}, source={cfg.source.kind}")
    print(f"  min wavelength (vs @ 2.5*f_peak) = {lam_min:.2f} m -> {lam_min / grid.dx:.1f} cells")
    if lam_min / grid.dx < 5:
        print("  WARNING: fewer than 5 cells per minimum wavelength, expect dispersion")

    if not args.no_figs:
        fig, ax = plt.subplots(figsize=(12, 7))
        fplot.plot_model(model, grid, src_pos, rcv_pos, names, ax=ax, title=f"{name}: vp")
        fig.savefig(out / "model_vp.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    t0 = time.perf_counter()
    rec = run_shots(cfg, grid, model, src_pos, rcv_pos)
    print(f"  propagation: {rec.seconds_per_shot:.2f} s/shot, inner dt {rec.inner_dt * 1e6:.2f} us "
          f"(x{rec.step_ratio}), total {time.perf_counter() - t0:.1f} s")

    if cfg.post.q or cfg.post.spreading_exponent:
        rec.data = post_apply(rec.data, rec.t, cfg.post.q, cfg.post.spreading_exponent)
        print(f"  post-processing: Q={cfg.post.q}, spreading exponent={cfg.post.spreading_exponent}")

    t0 = time.perf_counter()
    eik = eikonal_labels(grid, model, src_pos, rec.rcv_actual)
    print(f"  eikonal: {time.perf_counter() - t0:.1f} s")

    w_on = wavelet_onset(rec.wavelet, cfg.sim.dt, rec.peak_time, cfg.labels)
    onset = pick_gather(rec.data, rec.t, cfg.labels, eikonal=eik, w_onset=w_on)
    eik4 = np.concatenate([eik, eik[:, :, 1:2]], axis=2)          # primary label uses the x-node position
    delay = emergence_delay(onset, eik4, w_on)
    off = offsets(src_pos, rcv_pos)
    print(f"  wavelet onset rel. peak: {w_on * 1e3:.3f} ms; "
          f"median delay ({cfg.labels.component}): {np.nanmedian(delay[:, :, -1]) * 1e3:.3f} ms, "
          f"range {np.nanmin(delay[:, :, -1]) * 1e3:.3f}..{np.nanmax(delay[:, :, -1]) * 1e3:.3f} ms")

    extra = dict(torch_version=torch.__version__, deepwave_version=deepwave._version.__version__,
                 machine=platform.platform())
    h5 = save_run(out / "run.h5", cfg, grid, model, rec, src_pos, rcv_pos, names, eik, onset, w_on, delay, extra)
    print(f"  saved {h5} ({h5.stat().st_size / 1e6:.1f} MB)")

    if not args.no_figs:
        make_figures(out, cfg, grid, model, rec, src_pos, rcv_pos, names, eik, onset, w_on, delay, off)
    print("done")


def make_figures(out, cfg, grid, model, rec, src_pos, rcv_pos, names, eik, onset, w_on, delay, off, with_snapshots=True):
    n_rcv = rcv_pos.shape[0]
    tmax = min(rec.t[-1], np.nanmax(eik) + 6e-3)
    for r in range(n_rcv):
        fig, axes = plt.subplots(1, 3, figsize=(18, 0.35 * src_pos.shape[0] + 2), sharey=True)
        for ci, c in enumerate(COMPONENTS):
            picks = {"eikonal+w_onset": (eik[:, r, min(ci, 1)] + w_on, "tab:blue"),
                     f"onset ({cfg.labels.picker}, {cfg.labels.threshold_ref} ref, {cfg.labels.rel_threshold:g})": (onset[:, r, ci], "r")}
            fplot.plot_gather(rec.data[:, r, ci, :], rec.t, labels=off[:, r], picks=picks, ax=axes[ci],
                              tmin=rec.t[0], tmax=tmax, title=f"{names[r]} comp {c}", ylabel="offset [m]")
        fig.suptitle(f"{cfg.name}: receiver gather {names[r]} (rows = shots)")
        fig.savefig(out / f"gather_{names[r]}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    fplot.plot_decay(rec.data, rec.t, off, names, ax=ax, title=f"{cfg.name}: amplitude decay (comp x)")
    fig.savefig(out / "decay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    fplot.plot_delay_vs_offset(off, delay[:, :, -1], names, ax=ax,
                               title=f"{cfg.name}: emergence delay, component {cfg.labels.component}")
    fig.savefig(out / "delay_vs_offset.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    if cfg.sim.snapshots_ms and with_snapshots:
        ss = min(cfg.sim.snapshot_shot, src_pos.shape[0] - 1)
        snaps = snapshots(cfg, grid, model, src_pos, cfg.sim.snapshots_ms, ss)
        fig, axes = plt.subplots(len(snaps), 1, figsize=(12, 5 * len(snaps)))
        for ax, (tm, f) in zip(np.atleast_1d(axes), snaps.items()):
            fplot.plot_snapshot(f["vx"], grid, model, src_pos[ss:ss + 1],
                                rcv_pos, ax=ax, title=f"vx, t = {tm:.1f} ms after wavelet peak, shot {ss}")
        fig.savefig(out / "snapshots_vx.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
