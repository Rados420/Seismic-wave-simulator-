#!/usr/bin/env python
"""V1 dataset generator CLI (docs/dataset_design_v1.md).

    python generate.py sample  --seed 17                      # print the drawn parameters (no models, no simulation)
    python generate.py paint   --seed 17 --dims 2             # build both wall variants and plot them (no simulation)
    python generate.py run     --seeds 17 18 --dims 2         # simulate all wall x source variants, write HDF5
    python generate.py compare runs/gen/2d/draw_000017.h5 --rcv 0 --comp n

Common options: --meta my.yaml (overrides on top of fasim/gen/defaults.yaml),
--set path.to.key=value (repeatable, value parsed as YAML), --out runs/gen.
Example quick test: --set grid.dx_2d=0.2 --set acquisition.n_shots=4 --set record.length_ms=20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from fasim.gen.meta import load_meta
from fasim.gen.sampler import sample_draw, summary


def _common(p):
    p.add_argument("--meta", default=None, help="YAML with metaparameter overrides")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="key.path=value")
    p.add_argument("--out", default="runs/gen")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample", help="draw parameters only")
    _common(p)
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--json", action="store_true", help="print the full parameter dict")

    p = sub.add_parser("paint", help="build and plot the models of a draw (no simulation)")
    _common(p)
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--dims", type=int, choices=[2, 3], default=None)

    p = sub.add_parser("run", help="simulate draws")
    _common(p)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--seed-range", type=int, nargs=2, default=None, metavar=("FIRST", "STOP"))
    p.add_argument("--dims", type=int, choices=[2, 3], default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk", type=int, default=None, help="shots per Deepwave call")
    p.add_argument("--no-figs", action="store_true")

    p = sub.add_parser("compare", help="plot all variants of a simulated draw")
    p.add_argument("h5")
    p.add_argument("--rcv", type=int, nargs="+", default=[0])
    p.add_argument("--comp", default="n", help="p | s | n | v")
    p.add_argument("--tmax-ms", type=float, default=None)

    args = ap.parse_args()

    if args.cmd == "compare":
        from fasim.gen.plots import plot_compare
        for r in args.rcv:
            print(plot_compare(args.h5, r, args.comp, tmax_ms=args.tmax_ms))
        return

    meta = load_meta(args.meta, args.overrides)

    if args.cmd == "sample":
        for s in args.seed:
            prm = sample_draw(meta, s)
            print(json.dumps(prm, indent=1) if args.json else summary(prm))
            print()
        return

    dims = args.dims or int(meta["dims"])
    out = Path(args.out) / f"{dims}d"
    out.mkdir(parents=True, exist_ok=True)

    if args.cmd == "paint":
        from fasim.gen.build import build_variants
        from fasim.gen.plots import plot_models
        from fasim.gen.simulate import positions
        for s in args.seed:
            t0 = time.perf_counter()
            prm = sample_draw(meta, s)
            dom, geom, models = build_variants(prm, meta, dims, meta["enumerate"]["walls"])
            rcv, shots, _, _ = positions(prm, geom)
            path = out / f"draw_{s:06d}_models.png"
            plot_models(models, dom, geom, rcv, shots, prm, path, title=summary(prm))
            print(f"seed {s}: grid {dom.shape} ({dom.shape[0] * dom.shape[1] * (dom.shape[2] if dims == 3 else 1) / 1e6:.2f} M cells), "
                  f"built in {time.perf_counter() - t0:.1f} s -> {path}")
        return

    if args.cmd == "run":
        from fasim.gen.simulate import run_draw
        seeds = args.seeds or (list(range(*args.seed_range)) if args.seed_range else [0])
        for s in seeds:
            path = run_draw(meta, s, dims, out, device=args.device, chunk=args.chunk)
            if not args.no_figs:
                from fasim.gen.plots import plot_compare
                plot_compare(path, 0, "n")


if __name__ == "__main__":
    main()
