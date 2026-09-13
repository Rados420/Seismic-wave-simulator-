"""HDF5 persistence of a run: data, labels, geometry, model grids and the full config."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from .config import ExperimentCfg, from_dict, to_dict


def save_run(path: str | Path, cfg: ExperimentCfg, grid, model, rec, src_pos, rcv_pos, rcv_names,
             eikonal, onset, w_onset, delay, extra_attrs: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["config_json"] = json.dumps(to_dict(cfg))
        h.attrs["components"] = json.dumps(["y", "x", "p"])
        h.attrs["onset_components"] = json.dumps(["y", "x", "p", cfg.labels.component])
        h.attrs["wavelet_onset"] = float(w_onset)
        h.attrs["peak_time"] = float(rec.peak_time)
        h.attrs["inner_dt"] = float(rec.inner_dt)
        h.attrs["step_ratio"] = int(rec.step_ratio)
        h.attrs["seconds_per_shot"] = float(rec.seconds_per_shot)
        for k, v in (extra_attrs or {}).items():
            h.attrs[k] = str(v) if isinstance(v, str) else v
        h.create_dataset("data", data=rec.data, compression="gzip", compression_opts=4)
        h["t"] = rec.t
        h["wavelet"] = rec.wavelet
        h["src_pos"] = src_pos
        h["src_actual"] = rec.src_actual
        h["rcv_pos"] = rcv_pos
        h["rcv_actual"] = rec.rcv_actual
        h["rcv_names"] = np.array(rcv_names, dtype="S")
        h["offset"] = np.linalg.norm(src_pos[:, None, :] - rcv_pos[None, :, :], axis=-1)
        h["eikonal"] = eikonal
        h["onset"] = onset
        h["delay"] = delay
        g = h.create_group("model")
        g["vp"] = model.vp
        g["vs"] = model.vs
        g["rho"] = model.rho
        g["cavity"] = model.cavity
        g["edz"] = model.edz
        g["structure"] = model.structure
        g.attrs["z0"], g.attrs["x0"], g.attrs["dx"] = grid.z0, grid.x0, grid.dx
    return path


def load_run(path: str | Path) -> SimpleNamespace:
    from .geometry import Grid
    from .painter import Model
    with h5py.File(path, "r") as h:
        cfg = from_dict(json.loads(h.attrs["config_json"]))
        g = h["model"]
        model = Model(vp=g["vp"][()], vs=g["vs"][()], rho=g["rho"][()], cavity=g["cavity"][()].astype(bool),
                      edz=g["edz"][()].astype(bool), structure=g["structure"][()].astype(bool))
        grid = Grid(z0=float(g.attrs["z0"]), x0=float(g.attrs["x0"]), dx=float(g.attrs["dx"]),
                    nz=model.vp.shape[0], nx=model.vp.shape[1])
        ns = SimpleNamespace(
            path=str(path), cfg=cfg, grid=grid, model=model,
            data=h["data"][()], t=h["t"][()], wavelet=h["wavelet"][()],
            src_pos=h["src_pos"][()], src_actual=h["src_actual"][()],
            rcv_pos=h["rcv_pos"][()], rcv_actual=h["rcv_actual"][()],
            rcv_names=[s.decode() for s in h["rcv_names"][()]],
            offset=h["offset"][()], eikonal=h["eikonal"][()], onset=h["onset"][()], delay=h["delay"][()],
            attrs={k: (json.loads(v) if k.endswith("components") else v) for k, v in h.attrs.items()},
        )
    return ns
