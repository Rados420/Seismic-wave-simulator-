"""Overlapped generation: CPU worker processes prepare draws, the GPU simulates, a writer thread saves.

    prepare (N processes)  ->  simulate (main process, GPU)  ->  write (1 thread)

The GPU never waits for model building, eikonal labels or HDF5 compression as long as the workers
keep up. Up to `workers + prefetch` prepared draws are held in memory at once. Draws are processed
in the order their preparation finishes; output files and index lines carry the seed, so order
does not matter. A failing draw is logged to `failures.jsonl` and skipped.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

from .device import describe, resolve_device
from .simulate import prepare_draw, simulate_prepared, write_draw


def _prepare(meta: dict, seed: int, dims: int) -> dict:
    return prepare_draw(meta, seed, dims)


def _log_failure(out_dir: Path, seed: int, stage: str, exc: BaseException) -> None:
    with open(out_dir / "failures.jsonl", "a") as fh:
        fh.write(json.dumps({"seed": seed, "stage": stage, "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exception(exc)[-3:]}) + "\n")
    print(f"[seed {seed}] FAILED in {stage}: {type(exc).__name__}: {exc}", flush=True)


def run_pipeline(meta: dict, seeds, dims: int, out_dir, device: str = "auto", chunk: int | None = None,
                 workers: int = 2, prefetch: int = 2, skip_existing: bool = False, verbose: bool = True) -> list[Path]:
    device = resolve_device(device)
    desc = describe(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [s for s in seeds if not (skip_existing and (out_dir / f"draw_{s:06d}.h5").exists())]
    if verbose:
        print(f"pipeline: {len(seeds)} draws, {dims}-D, device {device} ({desc}), {workers} prepare workers, "
              f"prefetch {prefetch}", flush=True)
    t_start = time.perf_counter()
    gpu_busy = 0.0
    written, write_futs = [], {}
    pending = iter(seeds)
    ctx = mp.get_context("spawn")                     # no fork: CUDA and forked children do not mix
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx) as pool, \
            ThreadPoolExecutor(max_workers=1) as writer:
        prep_futs = {}

        def refill():
            while len(prep_futs) < max(1, workers) + prefetch:
                try:
                    s = next(pending)
                except StopIteration:
                    return
                prep_futs[pool.submit(_prepare, meta, s, dims)] = s

        refill()
        n_done = 0
        while prep_futs:
            done, _ = wait(prep_futs, return_when=FIRST_COMPLETED)
            for fut in done:
                seed = prep_futs.pop(fut)
                refill()                              # keep the workers busy while the GPU runs
                try:
                    prep = fut.result()
                except Exception as e:  # noqa: BLE001
                    _log_failure(out_dir, seed, "prepare", e)
                    continue
                try:
                    t0 = time.perf_counter()
                    results = simulate_prepared(prep, meta, device, chunk, verbose=False)
                    gpu_busy += time.perf_counter() - t0
                except Exception as e:  # noqa: BLE001
                    _log_failure(out_dir, seed, "simulate", e)
                    continue
                write_futs[writer.submit(write_draw, prep, results, meta, out_dir, desc)] = seed
                n_done += 1
                if verbose:
                    el = time.perf_counter() - t_start
                    print(f"[seed {seed}] {n_done}/{len(seeds)}: prepared {prep['t_prepare']:.1f} s, "
                          f"simulated {prep['t_simulate']:.1f} s; {el:.0f} s elapsed, "
                          f"propagation busy {100 * gpu_busy / el:.0f} %", flush=True)
        for fut, seed in write_futs.items():
            try:
                written.append(fut.result())
            except Exception as e:  # noqa: BLE001
                _log_failure(out_dir, seed, "write", e)
    el = time.perf_counter() - t_start
    if verbose and seeds:
        print(f"pipeline done: {len(written)}/{len(seeds)} draws in {el:.0f} s "
              f"({3600 * len(written) / el:.0f} draws/h), propagation busy {100 * gpu_busy / el:.0f} %", flush=True)
    return written
