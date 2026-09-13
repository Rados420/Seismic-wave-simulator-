#!/usr/bin/env python
"""Re-pick onsets of a saved run with different label settings and redraw the figures (no simulation).

    python repick.py runs/homogeneous --threshold-ref gather --rel-threshold 0.05 [--picker sta_lta] [--component p] [--suffix _gather]
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")

from fasim.io import load_run
from fasim.onset import pick_gather, wavelet_onset, emergence_delay
from fasim.geometry import offsets
from fasim.postprocess import apply as post_apply
from run_experiment import make_figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--picker", default=None)
    ap.add_argument("--threshold-ref", choices=["trace", "gather", "window"], default=None)
    ap.add_argument("--rel-threshold", type=float, default=None)
    ap.add_argument("--ref-window-ms", type=float, default=None)
    ap.add_argument("--component", choices=["y", "x", "p", "vec"], default=None)
    ap.add_argument("--search-before-ms", type=float, default=None)
    ap.add_argument("--q", type=float, default=None, help="apply constant-Q attenuation to the loaded data")
    ap.add_argument("--spreading", type=float, default=None, help="apply extra spreading exponent")
    ap.add_argument("--suffix", default="", help="write figures to <run_dir><suffix>/ instead of overwriting")
    ap.add_argument("--write-h5", action="store_true", help="overwrite onset/delay in run.h5 with the new picks")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    r = load_run(run_dir / "run.h5")
    changes = {k.replace("-", "_"): v for k, v in vars(args).items()
               if k in ("picker", "threshold_ref", "rel_threshold", "ref_window_ms", "component", "search_before_ms")
               and v is not None}
    lc = dataclasses.replace(r.cfg.labels, **changes)
    post = dataclasses.replace(r.cfg.post, **{k: v for k, v in dict(q=args.q, spreading_exponent=args.spreading).items() if v is not None})
    cfg = dataclasses.replace(r.cfg, labels=lc, post=post)
    if args.q is not None or args.spreading is not None:
        r.data = post_apply(r.data, r.t, post.q, post.spreading_exponent)
        print(f"post-processing applied: Q={post.q}, spreading exponent={post.spreading_exponent}")
    dt = float(r.t[1] - r.t[0])
    w_on = wavelet_onset(r.wavelet, dt, r.attrs["peak_time"], lc)
    onset = pick_gather(r.data, r.t, lc, eikonal=r.eikonal, w_onset=w_on)
    eik4 = np.concatenate([r.eikonal, r.eikonal[:, :, 1:2]], axis=2)
    delay = emergence_delay(onset, eik4, w_on)
    print(f"labels: {lc}")
    print(f"wavelet onset {w_on * 1e3:.3f} ms; median delay ({lc.component}) {np.nanmedian(delay[:, :, -1]) * 1e3:.3f} ms, "
          f"range {np.nanmin(delay[:, :, -1]) * 1e3:.3f}..{np.nanmax(delay[:, :, -1]) * 1e3:.3f} ms")

    rec = dataclasses.make_dataclass("Rec", ["data", "t", "wavelet", "peak_time"])(
        r.data, r.t, r.wavelet, r.attrs["peak_time"])
    out = Path(str(run_dir) + args.suffix)
    out.mkdir(parents=True, exist_ok=True)
    make_figures(out, cfg, r.grid, r.model, rec, r.src_pos, r.rcv_pos, r.rcv_names, r.eikonal, onset, w_on, delay,
                 offsets(r.src_pos, r.rcv_pos), with_snapshots=False)
    if (args.q is not None or args.spreading is not None) and args.suffix:
        from fasim.io import save_run
        rec_full = dataclasses.make_dataclass("RecFull", ["data", "t", "wavelet", "peak_time", "rcv_actual", "src_actual",
                                                          "inner_dt", "step_ratio", "seconds_per_shot"])(
            r.data, r.t, r.wavelet, r.attrs["peak_time"], r.rcv_actual, r.src_actual,
            r.attrs["inner_dt"], r.attrs["step_ratio"], r.attrs["seconds_per_shot"])
        save_run(out / "run.h5", cfg, r.grid, r.model, rec_full, r.src_pos, r.rcv_pos, r.rcv_names,
                 r.eikonal, onset, w_on, delay)
        print(f"processed data saved to {out / 'run.h5'}")
    if args.write_h5:
        with h5py.File(run_dir / "run.h5", "a") as h:
            h["onset"][...] = onset
            h["delay"][...] = delay
            h.attrs["wavelet_onset"] = float(w_on)
            import json
            from fasim.config import to_dict
            h.attrs["config_json"] = json.dumps(to_dict(cfg))
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
