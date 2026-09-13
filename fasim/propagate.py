"""Deepwave elastic wrapper: batched shots, hammer (force) or explosive (pressure) source."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import deepwave
from deepwave.common import cfl_condition_n, vpvsrho_to_lambmubuoyancy
from deepwave.wavelets import ricker

from .config import ExperimentCfg
from .geometry import Grid, force_direction
from .painter import Model

COMPONENTS = ("y", "x", "p")   # 0: v along tunnel, 1: v across tunnel, 2: pressure


@dataclass
class ShotRecord:
    data: np.ndarray          # [shot, rcv, comp, nt]
    t: np.ndarray             # [nt] time axis, 0 = source wavelet peak (see time_axis)
    wavelet: np.ndarray       # [nt] source time function as fed to Deepwave
    peak_time: float          # wavelet peak time on the Deepwave sample axis [s]
    rcv_actual: np.ndarray    # [rcv, comp, 2] physical positions of the staggered receiver nodes
    src_actual: np.ndarray    # [shot, 2] physical position of the source node (first component)
    inner_dt: float
    step_ratio: int
    seconds_per_shot: float


def make_wavelet(cfg: ExperimentCfg, nt: int) -> tuple[torch.Tensor, float]:
    f = cfg.source.peak_freq
    peak_time = 1.5 / f
    w = ricker(f, nt, cfg.sim.dt, peak_time) * cfg.source.amplitude
    return w, peak_time


def time_axis(nt: int, dt: float, peak_time: float, source_kind: str, receiver_kind: str) -> np.ndarray:
    """Physical time of each output sample relative to the wavelet peak.

    Deepwave puts force sources and velocity receivers at (k-0.5)*dt, pressure
    sources and pressure receivers at k*dt. When source and receiver sit on the
    same half of the staggered time grid the shift cancels.
    """
    src_half = 0.5 if source_kind == "force" else 0.0
    rcv_half = 0.5 if receiver_kind in ("y", "x") else 0.0
    k = np.arange(nt)
    return (k - rcv_half) * dt - (peak_time - src_half * dt)


def model_tensors(model: Model, device: str):
    vp = torch.tensor(model.vp, device=device)
    vs = torch.tensor(model.vs, device=device)
    rho = torch.tensor(model.rho, device=device)
    return vpvsrho_to_lambmubuoyancy(vp, vs, rho)


def _locations(idx: np.ndarray, n_shots: int, device: str) -> torch.Tensor:
    """idx (n, 2) -> [shot, n, 2] long tensor, replicated over shots."""
    t = torch.tensor(idx, dtype=torch.long, device=device)
    return t.unsqueeze(0).repeat(n_shots, 1, 1)


def run_shots(cfg: ExperimentCfg, grid: Grid, model: Model, src_pos: np.ndarray, rcv_pos: np.ndarray,
              verbose: bool = True) -> ShotRecord:
    sim, src = cfg.sim, cfg.source
    dev = sim.device
    nt = int(round(sim.record_ms * 1e-3 / sim.dt))
    wavelet, peak_time = make_wavelet(cfg, nt)
    lamb, mu, buoy = model_tensors(model, dev)

    # receivers: one staggered node set per component
    rcv_idx, rcv_actual = {}, []
    for c in COMPONENTS:
        idx, act = grid.to_index(rcv_pos, c)
        rcv_idx[c] = idx
        rcv_actual.append(act)
    rcv_actual = np.stack(rcv_actual, axis=1)            # [rcv, comp, 2]

    # sources
    n_shots = src_pos.shape[0]
    if src.kind == "force":
        d = force_direction(cfg.acquisition, src.direction_deg)   # (dz, dx)
        src_idx_y, act_y = grid.to_index(src_pos, "y")
        src_idx_x, act_x = grid.to_index(src_pos, "x")
        src_actual = act_y if abs(d[0]) >= abs(d[1]) else act_x
    elif src.kind == "pressure":
        src_idx_p, src_actual = grid.to_index(src_pos, "p")
    else:
        raise ValueError(f"source.kind must be force|pressure, got {src.kind!r}")

    vmax = float(max(model.vp.max(), 1.0))
    inner_dt, step_ratio = cfl_condition_n([grid.dx, grid.dx], sim.dt, vmax)

    n_rcv = rcv_pos.shape[0]
    data = np.zeros((n_shots, n_rcv, len(COMPONENTS), nt), dtype=np.float32)
    t_start = time.perf_counter()
    for s0 in range(0, n_shots, sim.shot_chunk):
        sl = slice(s0, min(s0 + sim.shot_chunk, n_shots))
        ns = sl.stop - sl.start
        amp = wavelet.to(dev).reshape(1, 1, nt).repeat(ns, 1, 1)
        kw = dict(
            receiver_locations_y=_locations(rcv_idx["y"], ns, dev),
            receiver_locations_x=_locations(rcv_idx["x"], ns, dev),
            receiver_locations_p=_locations(rcv_idx["p"], ns, dev),
            pml_width=cfg.domain.pml_width,
            pml_freq=src.peak_freq,
            accuracy=sim.accuracy,
            max_vel=vmax,
        )
        if src.kind == "force":
            if abs(d[0]) > 1e-9:
                kw["source_amplitudes_y"] = amp * float(d[0])
                kw["source_locations_y"] = torch.tensor(src_idx_y[sl], dtype=torch.long, device=dev).unsqueeze(1)
            if abs(d[1]) > 1e-9:
                kw["source_amplitudes_x"] = amp * float(d[1])
                kw["source_locations_x"] = torch.tensor(src_idx_x[sl], dtype=torch.long, device=dev).unsqueeze(1)
        else:
            kw["source_amplitudes_p"] = amp
            kw["source_locations_p"] = torch.tensor(src_idx_p[sl], dtype=torch.long, device=dev).unsqueeze(1)

        with torch.no_grad():
            out = deepwave.elastic(lamb, mu, buoy, grid.dx, sim.dt, **kw)
        rec_p, rec_y, rec_x = out[-3], out[-2], out[-1]        # [shot, rcv, nt] (2-D: p, vy, vx)
        data[sl, :, 0, :] = rec_y.cpu().numpy()
        data[sl, :, 1, :] = rec_x.cpu().numpy()
        data[sl, :, 2, :] = rec_p.cpu().numpy()
        if verbose:
            el = time.perf_counter() - t_start
            print(f"  shots {sl.start:3d}-{sl.stop - 1:3d} done, {el:6.1f} s elapsed "
                  f"({el / sl.stop:.2f} s/shot)", flush=True)

    sps = (time.perf_counter() - t_start) / n_shots
    # time axis: components y/x are velocities, p is pressure -> store the velocity axis,
    # pressure samples are dt/2 later than the stored axis says when the source is a force.
    t = time_axis(nt, sim.dt, peak_time, src.kind, "x")
    return ShotRecord(data=data, t=t, wavelet=wavelet.numpy(), peak_time=peak_time,
                      rcv_actual=rcv_actual, src_actual=src_actual,
                      inner_dt=inner_dt, step_ratio=step_ratio, seconds_per_shot=sps)


def snapshots(cfg: ExperimentCfg, grid: Grid, model: Model, src_pos: np.ndarray, times_ms: list[float],
              shot: int = 0) -> dict[float, dict[str, np.ndarray]]:
    """Wavefield snapshots (vy, vx) for one shot at the requested times (relative to wavelet peak).

    Implemented by re-running the propagation up to each time and taking the
    final wavefield; cheap enough for a handful of snapshot times.
    """
    sim, src = cfg.sim, cfg.source
    dev = sim.device
    nt_full = int(round(sim.record_ms * 1e-3 / sim.dt))
    wavelet, peak_time = make_wavelet(cfg, nt_full)
    lamb, mu, buoy = model_tensors(model, dev)
    vmax = float(max(model.vp.max(), 1.0))
    pos = src_pos[shot:shot + 1]
    out_dict = {}
    for tm in times_ms:
        nt = int(round((tm * 1e-3 + peak_time) / sim.dt))
        nt = max(2, min(nt, nt_full))
        amp = wavelet[:nt].to(dev).reshape(1, 1, nt)
        # Deepwave needs at least one receiver to size the batch: record at the source node
        dummy_idx, _ = grid.to_index(pos, "x")
        kw = dict(pml_width=cfg.domain.pml_width, pml_freq=src.peak_freq, accuracy=sim.accuracy, max_vel=vmax,
                  receiver_locations_x=torch.tensor(dummy_idx, dtype=torch.long, device=dev).unsqueeze(1))
        if src.kind == "force":
            d = force_direction(cfg.acquisition, src.direction_deg)
            iy, _ = grid.to_index(pos, "y")
            ix, _ = grid.to_index(pos, "x")
            if abs(d[0]) > 1e-9:
                kw["source_amplitudes_y"] = amp * float(d[0])
                kw["source_locations_y"] = torch.tensor(iy, dtype=torch.long, device=dev).unsqueeze(1)
            if abs(d[1]) > 1e-9:
                kw["source_amplitudes_x"] = amp * float(d[1])
                kw["source_locations_x"] = torch.tensor(ix, dtype=torch.long, device=dev).unsqueeze(1)
        else:
            ip, _ = grid.to_index(pos, "p")
            kw["source_amplitudes_p"] = amp
            kw["source_locations_p"] = torch.tensor(ip, dtype=torch.long, device=dev).unsqueeze(1)
        with torch.no_grad():
            out = deepwave.elastic(lamb, mu, buoy, grid.dx, sim.dt, **kw)
        vy, vx = out[0][0].cpu().numpy(), out[1][0].cpu().numpy()
        out_dict[tm] = {"vy": _crop(vy, grid.shape), "vx": _crop(vx, grid.shape)}
    return out_dict


def _crop(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Deepwave returns wavefields including PML padding; centre-crop to the model."""
    if field.shape == shape:
        return field
    oz = (field.shape[0] - shape[0]) // 2
    ox = (field.shape[1] - shape[1]) // 2
    return field[oz:oz + shape[0], ox:ox + shape[1]]
