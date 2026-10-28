#!/usr/bin/env python
"""V1 dataset generator CLI (FA_Magic/docs/dataset_design_v1.md).

    python generate.py sample  --seed 17                      # print the drawn parameters (no models, no simulation)
    python generate.py paint   --seed 17 --dims 2             # build both wall variants and plot them (no simulation)
    python generate.py run     --seeds 17 18 --dims 2         # simulate all wall x source variants, write HDF5
    python generate.py compare runs/gen/2d/draw_000017.h5 --rcv 0 --comp n
    python generate.py gpu-info                               # what torch sees; tiny GPU-vs-CPU Deepwave check
    python generate.py bench   --seed 0 --dims 2              # time one draw's propagation (bare wall, both sources)

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
    p.add_argument("--device", default=None, help="auto | cuda | cuda:N | cpu (default: sim.device)")
    p.add_argument("--chunk", type=int, default=None, help="shots per Deepwave call (default: fit to GPU memory)")
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--workers", type=int, default=None,
                   help="CPU processes preparing draws in parallel with the GPU (default: sim.workers; 0 = sequential)")
    p.add_argument("--prefetch", type=int, default=None, help="extra prepared draws held ready (default: sim.prefetch)")
    p.add_argument("--skip-existing", action="store_true", help="skip seeds whose output file already exists")

    p = sub.add_parser("gpu-info", help="check the CUDA / Deepwave set-up")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("bench", help="time the propagation of one draw (bare wall, all sources)")
    _common(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dims", type=int, choices=[2, 3], default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--compare-cpu", type=int, default=0, metavar="N",
                   help="also run the first N shots on the CPU and report the difference")

    p = sub.add_parser("compare", help="plot all variants of a simulated draw")
    p.add_argument("h5")
    p.add_argument("--rcv", type=int, nargs="+", default=[0])
    p.add_argument("--comp", default="n", help="p | s | n | v")
    p.add_argument("--tmax-ms", type=float, default=None)

    args = ap.parse_args()

    if args.cmd == "gpu-info":
        from fasim.gen.device import environment_check
        raise SystemExit(0 if environment_check(args.device) else 1)

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

    if args.cmd == "bench":
        bench(meta, args.seed, dims, args.device or meta["sim"]["device"], args.chunk, args.compare_cpu)
        return

    if args.cmd == "run":
        seeds = args.seeds or (list(range(*args.seed_range)) if args.seed_range else [0])
        device = args.device or meta["sim"]["device"]
        workers = meta["sim"]["workers"] if args.workers is None else args.workers
        prefetch = meta["sim"]["prefetch"] if args.prefetch is None else args.prefetch
        if workers > 0:
            from fasim.gen.pipeline import run_pipeline
            paths = run_pipeline(meta, seeds, dims, out, device, args.chunk, workers, prefetch, args.skip_existing)
        else:
            from fasim.gen.simulate import run_draw
            paths = [run_draw(meta, s, dims, out, device=device, chunk=args.chunk) for s in seeds
                     if not (args.skip_existing and (out / f"draw_{s:06d}.h5").exists())]
        if not args.no_figs:
            from fasim.gen.plots import plot_compare
            for path in paths:
                plot_compare(path, 0, "n")


def bench(meta, seed, dims, device, chunk, compare_cpu):
    """Propagation timing of one draw on the bare-wall model, all source types batched together."""
    import numpy as np
    import torch
    from fasim.gen.build import build_variants
    from fasim.gen.device import auto_chunk, bytes_per_shot, describe, is_cuda, padded_cells, resolve_device
    from fasim.gen.simulate import (SOURCE_KIND, frequency_cap, impact_directions, nudge_into_rock, positions,
                                    propagate_batch)
    device = resolve_device(device)
    prm = sample_draw(meta, seed)
    t0 = time.perf_counter()
    dom, geom, models = build_variants(prm, meta, dims, ["bare"])
    m = models["bare"]
    rcv, shots, _, heights = positions(prm, geom)
    blocked = m["excavated"] | (m["labels"] == 6)
    nudge_into_rock(rcv, blocked, dom, geom, None)
    nudge_into_rock(shots, blocked, dom, geom, heights)
    t_build = time.perf_counter() - t0
    cap = frequency_cap(models, meta, dom.dx)
    jobs = [{"name": s, "kind": SOURCE_KIND[s], "freq": min(prm["sources"][s]["peak_freq"], cap),
             "amplitude": prm["sources"][s]["amplitude"], "directions": impact_directions(prm, geom, dims)}
            for s in meta["enumerate"]["sources"]]
    n_total = len(shots) * len(jobs)
    ch = chunk or auto_chunk(dom.shape, meta, device, n_total)
    print(f"device   : {device}: {describe(device)}")
    print(f"grid     : {dims}-D {dom.shape} @ {dom.dx} m = {np.prod(dom.shape) / 1e6:.2f} M cells "
          f"({padded_cells(dom.shape, meta) / 1e6:.2f} M with PML), model build {t_build:.1f} s")
    print(f"batch    : {n_total} shots ({len(shots)} x {len(jobs)} sources), {ch} per call, "
          f"estimated {bytes_per_shot(dom.shape, meta) / 2**20:.0f} MB per shot")
    if is_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)
    res = propagate_batch(m, dom, shots, rcv, jobs, meta, device, ch, verbose=True)
    sps = next(iter(res.values()))[3]
    print(f"result   : {sps:.3f} s per shot, {sps * n_total:.1f} s for the batch, "
          f"{sps * n_total * len(meta['enumerate']['walls']):.1f} s per draw (all walls)")
    if is_cuda(device):
        peak = torch.cuda.max_memory_allocated(device)
        print(f"memory   : peak {peak / 2**30:.2f} GB, {peak / min(ch, n_total) / 2**20:.0f} MB per shot in the batch")
    if compare_cpu:
        k = min(compare_cpu, len(shots))
        ref = propagate_batch(m, dom, shots[:k], rcv, [dict(j, directions=j["directions"][:k]) for j in jobs],
                              meta, "cpu", None, verbose=False)
        for j in jobs:
            a, b = res[j["name"]][0][:k], ref[j["name"]][0]
            err = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))
            print(f"vs cpu   : {j['name']}: max relative difference {err:.1e} over {k} shots")


if __name__ == "__main__":
    main()
