"""Simulate one draw: all enumerated (wall x source) variants, eikonal labels, HDF5 output."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import skfmm
import torch
import deepwave
from deepwave.common import vpvsrho_to_lambmubuoyancy
from deepwave.wavelets import ricker
from scipy.ndimage import map_coordinates

from ..propagate import time_axis
from .build import Domain, TunnelGeom, build_variants
from .device import auto_chunk, describe, resolve_device, sync
from .sampler import sample_draw

SOURCE_KIND = {"explosive": "pressure", "impact": "force"}
DW_NAMES = ("z", "y", "x")


# --------------------------------------------------------------------------- #
def positions(params: dict, geom: TunnelGeom):
    """Physical positions of receivers (real then virtual) and shots, plus the virtual mask."""
    acq = params["acquisition"]
    rp = [geom.wall_point(p["s"], p["embed"], p["height"]) for p in acq["receivers"] + acq["virtual"]]
    sp = [geom.wall_point(p["s"], p["embed"], p["height"]) for p in acq["shots"]]
    is_virtual = np.array([False] * len(acq["receivers"]) + [True] * len(acq["virtual"]))
    heights = [p["height"] for p in acq["shots"]]
    return np.array(rp), np.array(sp), is_virtual, heights


def nudge_into_rock(pos: np.ndarray, blocked: np.ndarray, dom: Domain, geom: TunnelGeom, heights) -> np.ndarray:
    """Move points that fall into the cavity / a void outward along the wall normal (max 2 m). Returns shifts."""
    shifts = np.zeros(len(pos))
    for i in range(len(pos)):
        step = geom.outward(heights[i] if heights is not None else 0.0) * dom.dx
        for k in range(int(2.0 / dom.dx)):
            idx, _ = dom.to_index(pos[i])
            if not blocked[tuple(idx[0])]:
                break
            pos[i] += step
            shifts[i] += dom.dx
    return shifts


def impact_directions(params: dict, geom: TunnelGeom, dims: int) -> np.ndarray:
    """Unit force vectors per shot: wall normal tilted by (polar, azimuth); 2-D = in-plane projection."""
    im = params["sources"]["impact"]
    out = []
    for p, pol, az in zip(params["acquisition"]["shots"], im["polar_deg"], im["azimuth_deg"]):
        th = geom.theta(p["height"])
        n_hat = np.array([0.0, np.cos(th), np.sin(th)])
        s_hat = np.array([1.0, 0.0, 0.0])
        t_hat = np.cross(n_hat, s_hat)               # tangential, around the tunnel
        a, b = np.deg2rad(pol), np.deg2rad(az)
        d = np.cos(a) * n_hat + np.sin(a) * (np.cos(b) * s_hat + np.sin(b) * t_hat)
        d = d[:dims]
        out.append(d / (np.linalg.norm(d) + 1e-12))
    return np.array(out)


def frequency_cap(models: dict, meta: dict, dx: float) -> float:
    """Highest Ricker peak frequency the grid supports (vs at the 0.5th percentile of solid cells)."""
    vs_all = np.concatenate([m["vs"][m["vs"] > 0].ravel() for m in models.values()])
    vs_ref = float(np.percentile(vs_all, 0.5))
    return vs_ref / (meta["grid"]["min_cells_per_wavelength"] * dx * 2.5)


# --------------------------------------------------------------------------- #
def propagate_batch(model: dict, dom: Domain, shot_pos: np.ndarray, rcv_pos: np.ndarray, jobs: list[dict],
                    meta: dict, device: str, chunk: int | None = None, verbose: bool = True) -> dict:
    """Deepwave elastic for several source jobs on one model, batched into the same calls.

    jobs: [{"name", "kind" ("pressure" | "force"), "freq", "amplitude", "directions" (force only)}].
    Every job uses all shot positions; job j, shot k is batch entry j * n_shot + k. In each call a shot
    carries a pressure and a force source, one of them with zero amplitude, so explosive and impact
    shots share one call (fills the GPU better for small 2-D grids). Deepwave takes one PML frequency
    per call: the mean of the jobs' peak frequencies.

    Returns {name: (data [shot, rcv, comp, nt] with comps (p, s, n[, v]), t, wavelet, seconds per shot)}.
    """
    dims = dom.dims
    dt = float(meta["record"]["dt"])
    nt = int(round(meta["record"]["length_ms"] * 1e-3 / dt))
    names = DW_NAMES[-dims:]
    n_shot, n_rcv = len(shot_pos), len(rcv_pos)
    n_total = n_shot * len(jobs)

    # Deepwave sources are densities (force / volume, pressure rate per cell): divide by the cell
    # volume so the physical source strength, and the recorded amplitudes, do not depend on dx
    wavs = [ricker(j["freq"], nt, dt, 1.5 / j["freq"]) * j["amplitude"] / dom.dx ** dims for j in jobs]
    W = torch.stack(wavs).to(device)                                   # [job, nt]
    p_on = torch.tensor([1.0 if j["kind"] == "pressure" else 0.0 for j in jobs], device=device)
    F = np.zeros((len(jobs), n_shot, dims), np.float32)                # force direction per job / shot
    for i, j in enumerate(jobs):
        if j["kind"] == "force":
            F[i] = j["directions"]
    F = torch.tensor(F, device=device)
    any_p = bool(p_on.any())
    any_f = bool(F.abs().sum() > 0)

    vp, vs, rho = (torch.tensor(model[k], device=device) for k in ("vp", "vs", "rho"))
    lamb, mu, buoy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    vmax = float(model["vp"].max()) * 1.001     # margin: Deepwave recomputes vp from lambda, mu, rho in float32
    pml_freq = float(np.mean([j["freq"] for j in jobs]))

    def loc(idx):
        return torch.tensor(idx, dtype=torch.long, device=device)

    rcv_idx = {None: loc(dom.to_index(rcv_pos)[0])}
    rcv_idx.update({d: loc(dom.to_index(rcv_pos, d)[0]) for d in range(dims)})
    src_idx = {None: loc(dom.to_index(shot_pos)[0])}
    src_idx.update({d: loc(dom.to_index(shot_pos, d)[0]) for d in range(dims)})

    chunk = chunk or auto_chunk(dom.shape, meta, device, n_total)
    out_data = np.zeros((len(jobs), n_shot, n_rcv, dims + 1, nt), np.float32)
    sync(device)
    t0 = time.perf_counter()
    a = 0
    while a < n_total:
        b = min(a + chunk, n_total)
        g = torch.arange(a, b, device=device)
        jj, kk = g // n_shot, g % n_shot                                 # job and shot of each batch entry
        ns = b - a
        kw = {"receiver_locations_p": rcv_idx[None].unsqueeze(0).expand(ns, -1, -1).contiguous()}
        for d in range(dims):
            kw[f"receiver_locations_{names[d]}"] = rcv_idx[d].unsqueeze(0).expand(ns, -1, -1).contiguous()
        wav = W[jj]                                                      # [ns, nt]
        if any_p:
            kw["source_amplitudes_p"] = (wav * p_on[jj, None]).unsqueeze(1)
            kw["source_locations_p"] = src_idx[None][kk].unsqueeze(1)
        if any_f:
            for d in range(dims):
                kw[f"source_amplitudes_{names[d]}"] = (wav * F[jj, kk, d][:, None]).unsqueeze(1)
                kw[f"source_locations_{names[d]}"] = src_idx[d][kk].unsqueeze(1)
        try:
            with torch.no_grad():
                out = deepwave.elastic(lamb, mu, buoy, dom.dx, dt, pml_width=meta["grid"]["pml_width"],
                                       pml_freq=pml_freq, accuracy=meta["grid"]["accuracy"], max_vel=vmax, **kw)
        except torch.cuda.OutOfMemoryError:
            if chunk == 1:
                raise
            del kw
            torch.cuda.empty_cache()
            chunk = max(1, chunk // 2)
            if verbose:
                print(f"      out of GPU memory, retrying with {chunk} shots per call", flush=True)
            continue
        rec = out[-(dims + 1):]                                          # p, then velocity along dim 0..dims-1
        jn, kn = jj.cpu().numpy(), kk.cpu().numpy()
        for c, rr in enumerate(rec):
            out_data[jn, kn, :, c, :] = rr.cpu().numpy()
        del out, rec
        if verbose:
            el = time.perf_counter() - t0
            print(f"      shots {a}-{b - 1} of {n_total} ({chunk} per call): {el:.1f} s ({el / b:.2f} s/shot)",
                  flush=True)
        a = b
    sync(device)
    sps = (time.perf_counter() - t0) / n_total
    res = {}
    for i, j in enumerate(jobs):
        t = time_axis(nt, dt, 1.5 / j["freq"], j["kind"], "x")         # velocity comps; pressure lags dt/2 for force
        res[j["name"]] = (out_data[i], t, wavs[i].numpy(), sps)
    return res


def propagate(model: dict, dom: Domain, shot_pos: np.ndarray, rcv_pos: np.ndarray, kind: str, freq: float,
              amplitude: float, directions: np.ndarray | None, meta: dict, device: str, chunk: int | None,
              verbose: bool = True):
    """Single source type. Returns (data [shot, rcv, comp, nt], t, wavelet, seconds per shot)."""
    job = {"name": "src", "kind": kind, "freq": freq, "amplitude": amplitude, "directions": directions}
    return propagate_batch(model, dom, shot_pos, rcv_pos, [job], meta, device, chunk, verbose)["src"]


def eikonal(model: dict, dom: Domain, src: np.ndarray, rcv: np.ndarray) -> np.ndarray:
    """First-arrival P traveltime [shot, rcv]; fast marching from the smaller of the two point sets."""
    from_rcv = len(rcv) < len(src)
    a, b = (rcv, src) if from_rcv else (src, rcv)
    speed = model["vp"].astype(np.float64)
    mask = model["cavity"] | (speed <= 0)
    S, N, V = dom.coords()
    out = np.zeros((len(a), len(b)))
    for i, p in enumerate(a):
        dist = (S - p[0]) ** 2 + (N - p[1]) ** 2 + ((V - p[2]) ** 2 if dom.dims == 3 else 0.0)
        phi = np.sqrt(np.broadcast_to(dist, dom.shape)) - dom.dx
        idx = tuple(dom.to_index(p)[0][0])
        v0 = speed[idx] if speed[idx] > 0 else np.median(speed[speed > 0])
        tt = skfmm.travel_time(np.ma.MaskedArray(phi, mask), np.ma.MaskedArray(speed, mask), dx=dom.dx, order=2)
        tt = np.ma.filled(tt, np.nan) + dom.dx / v0
        tt[phi < 0] = (phi[phi < 0] + dom.dx) / v0
        fi = ((b - dom.origin) / dom.dx).T
        out[i] = map_coordinates(np.nan_to_num(tt, nan=np.nanmax(tt)), fi, order=1, mode="nearest")
    return out.T if from_rcv else out


# --------------------------------------------------------------------------- #
def run_draw(meta: dict, seed: int, dims: int, out_dir: str | Path, device: str = "auto",
             chunk: int | None = None, verbose: bool = True) -> Path:
    t_start = time.perf_counter()
    device = resolve_device(device)
    params = sample_draw(meta, seed)
    walls, sources = meta["enumerate"]["walls"], meta["enumerate"]["sources"]
    dom, geom, models = build_variants(params, meta, dims, walls)
    rcv, shots, is_virtual, heights = positions(params, geom)
    blocked = models[walls[0]]["excavated"] | (models[walls[0]]["labels"] == 6)
    rcv_shift = nudge_into_rock(rcv, blocked, dom, geom, [p["height"] for p in
                                params["acquisition"]["receivers"] + params["acquisition"]["virtual"]])
    shot_shift = nudge_into_rock(shots, blocked, dom, geom, heights)
    dirs = impact_directions(params, geom, dims)
    cap = frequency_cap(models, meta, dom.dx)
    if verbose:
        print(f"[seed {seed}] {dims}-D grid {dom.shape} @ {dom.dx} m ({np.prod(dom.shape) / 1e6:.2f} M cells), "
              f"{len(shots)} shots, {len(rcv)} receivers, frequency cap {cap:.0f} Hz, device {device}", flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"draw_{seed:06d}.h5"
    st = meta["storage"]
    dtype = np.float16 if st["dtype"] == "float16" else np.float32
    comp = dict(compression="gzip", compression_opts=st["compression"]) if st["compression"] else {}
    freq_used = {}
    with h5py.File(path, "w") as h:
        h.attrs.update(seed=seed, dims=dims, params_json=json.dumps(params), meta_json=json.dumps(meta),
                       components=json.dumps(["p", "s", "n", "v"][:dims + 1]), torch=str(torch.__version__),
                       deepwave=str(deepwave._version.__version__), machine=str(platform.platform()),
                       device=str(describe(device)))
        g = h.create_group("geometry")
        g["origin"], g["dx"], g["shape"] = dom.origin, dom.dx, np.array(dom.shape)
        g["rcv_pos"], g["rcv_is_virtual"], g["rcv_shift"] = rcv, is_virtual, rcv_shift
        g["shot_pos"], g["shot_shift"], g["impact_dir"] = shots, shot_shift, dirs
        g["offset"] = np.linalg.norm(shots[:, None, :] - rcv[None, :, :], axis=-1)
        for w in walls:
            m = models[w]
            gw = h.create_group(f"wall_{w}")
            if st["models"]:
                for k in ("vp", "vs", "rho"):
                    gw.create_dataset(k, data=m[k].astype(np.float16), **comp)
                gw.create_dataset("labels", data=m["labels"], **comp)
            if meta["labels"]["eikonal"]:
                te = time.perf_counter()
                gw["eikonal"] = eikonal(m, dom, shots, rcv)
                if verbose:
                    print(f"  wall {w}: eikonal {time.perf_counter() - te:.1f} s", flush=True)
            jobs = []
            for src in sources:
                sp = params["sources"][src]
                freq_used[src] = min(sp["peak_freq"], cap)
                jobs.append({"name": src, "kind": SOURCE_KIND[src], "freq": freq_used[src],
                             "amplitude": sp["amplitude"], "directions": dirs})
            groups = [jobs] if meta.get("sim", {}).get("combine_sources", True) else [[j] for j in jobs]
            results = {}
            for grp in groups:
                if verbose:
                    print(f"  wall {w}, sources " + ", ".join(f"{j['name']} {j['freq']:.0f} Hz" for j in grp), flush=True)
                results.update(propagate_batch(m, dom, shots, rcv, grp, meta, device, chunk, verbose))
            for src in sources:
                sp = params["sources"][src]
                f = freq_used[src]
                data, t, wav, sps = results[src]
                gs = gw.create_group(src)
                gs.create_dataset("data", data=data.astype(dtype), **comp)
                gs["t"], gs["wavelet"] = t, wav
                gs.attrs.update(peak_freq=f, peak_freq_requested=sp["peak_freq"], kind=SOURCE_KIND[src],
                                seconds_per_shot=sps)
        h.attrs["runtime_s"] = time.perf_counter() - t_start

    a = params["acquisition"]
    entry = {"seed": seed, "dims": dims, "file": path.name, "lithology": params["rock"]["host"]["name"],
             "quality": round(params["rock"]["quality"], 3),
             "structures": [s["type"] for s in params["rock"]["structures"]],
             "curvature_h": params["tunnel"]["curvature_horizontal"], "curvature_v": params["tunnel"]["curvature_vertical"],
             "n_receivers": a["n_receivers"], "gap": round(a["gap"], 2), "shot_spacing": round(a["shot_spacing"], 3),
             "freq": {k: round(v, 1) for k, v in freq_used.items()}, "runtime_s": round(time.perf_counter() - t_start, 1)}
    with open(out_dir / "index.jsonl", "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    if verbose:
        print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB) in {entry['runtime_s']} s", flush=True)
    return path
