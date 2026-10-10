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
def propagate(model: dict, dom: Domain, shot_pos: np.ndarray, rcv_pos: np.ndarray, kind: str, freq: float,
              amplitude: float, directions: np.ndarray | None, meta: dict, device: str, chunk: int,
              verbose: bool = True):
    """Deepwave elastic, all shots. Returns data [shot, rcv, comp, nt] with comps (p, s, n[, v])."""
    dims = dom.dims
    dt = float(meta["record"]["dt"])
    nt = int(round(meta["record"]["length_ms"] * 1e-3 / dt))
    peak_time = 1.5 / freq
    # Deepwave sources are densities (force / volume, pressure rate per cell): divide by the cell
    # volume so the physical source strength, and the recorded amplitudes, do not depend on dx
    wav = ricker(freq, nt, dt, peak_time) * amplitude / dom.dx ** dims
    vp, vs, rho = (torch.tensor(model[k], device=device) for k in ("vp", "vs", "rho"))
    lamb, mu, buoy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    names = DW_NAMES[-dims:]
    vmax = float(model["vp"].max())

    rcv_idx = {None: dom.to_index(rcv_pos)[0]}
    rcv_idx.update({d: dom.to_index(rcv_pos, d)[0] for d in range(dims)})
    n_shot, n_rcv = len(shot_pos), len(rcv_pos)
    data = np.zeros((n_shot, n_rcv, dims + 1, nt), np.float32)
    t0 = time.perf_counter()
    for a in range(0, n_shot, chunk):
        sl = slice(a, min(a + chunk, n_shot))
        ns = sl.stop - sl.start

        def loc(idx):
            return torch.tensor(idx, dtype=torch.long, device=device)

        kw = {"receiver_locations_p": loc(rcv_idx[None]).unsqueeze(0).repeat(ns, 1, 1)}
        for d in range(dims):
            kw[f"receiver_locations_{names[d]}"] = loc(rcv_idx[d]).unsqueeze(0).repeat(ns, 1, 1)
        amp = wav.to(device).reshape(1, 1, nt).repeat(ns, 1, 1)
        if kind == "pressure":
            kw["source_amplitudes_p"] = amp
            kw["source_locations_p"] = loc(dom.to_index(shot_pos[sl])[0]).unsqueeze(1)
        else:
            dirs = torch.tensor(directions[sl], dtype=amp.dtype, device=device)
            for d in range(dims):
                kw[f"source_amplitudes_{names[d]}"] = amp * dirs[:, d].reshape(ns, 1, 1)
                kw[f"source_locations_{names[d]}"] = loc(dom.to_index(shot_pos[sl], d)[0]).unsqueeze(1)
        with torch.no_grad():
            out = deepwave.elastic(lamb, mu, buoy, dom.dx, dt, pml_width=meta["grid"]["pml_width"],
                                   pml_freq=freq, accuracy=meta["grid"]["accuracy"], max_vel=vmax, **kw)
        rec = out[-(dims + 1):]                                   # p, then velocity along dim 0..dims-1
        for c, rr in enumerate(rec):
            data[sl, :, c, :] = rr.cpu().numpy()
        if verbose:
            el = time.perf_counter() - t0
            print(f"      shots {sl.start}-{sl.stop - 1}: {el:.1f} s ({el / sl.stop:.2f} s/shot)", flush=True)
    t = time_axis(nt, dt, peak_time, kind, "x")                  # velocity components; pressure lags dt/2 for force
    return data, t, wav.numpy(), (time.perf_counter() - t0) / n_shot


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
def run_draw(meta: dict, seed: int, dims: int, out_dir: str | Path, device: str = "cpu",
             chunk: int | None = None, verbose: bool = True) -> Path:
    t_start = time.perf_counter()
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
    chunk = chunk or (8 if dims == 2 else 2)
    if verbose:
        print(f"[seed {seed}] {dims}-D grid {dom.shape} @ {dom.dx} m ({np.prod(dom.shape) / 1e6:.2f} M cells), "
              f"{len(shots)} shots, {len(rcv)} receivers, frequency cap {cap:.0f} Hz", flush=True)

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
                       deepwave=str(deepwave._version.__version__), machine=str(platform.platform()))
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
            for src in sources:
                sp = params["sources"][src]
                f = min(sp["peak_freq"], cap)
                freq_used[src] = f
                if verbose:
                    print(f"  wall {w}, source {src}: {f:.0f} Hz", flush=True)
                data, t, wav, sps = propagate(m, dom, shots, rcv, SOURCE_KIND[src], f, sp["amplitude"],
                                              dirs, meta, device, chunk, verbose)
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
