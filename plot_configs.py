#!/usr/bin/env python
"""Paint every config in configs/ and show the vp (and optionally vs) maps side by side on one colour scale.

    python plot_configs.py [--field vp|vs|rho] [--out runs/velocity_maps.png] [configs/*.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fasim.config import load_config
from fasim.geometry import Grid, receiver_positions, shot_positions
from fasim.painter import paint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=None)
    ap.add_argument("--field", default="vp", choices=["vp", "vs", "rho"])
    ap.add_argument("--out", default="runs/velocity_maps.png")
    ap.add_argument("--no-tunnel-row", action="store_true", help="also show each config without the cavity")
    args = ap.parse_args()
    paths = [Path(p) for p in args.configs] if args.configs else sorted(Path("configs").glob("*.yaml"))

    models = []
    for p in paths:
        cfg = load_config(p)
        grid = Grid.from_domain(cfg.domain)
        m = paint(cfg, grid, np.random.default_rng(cfg.seed))
        models.append((p.stem, cfg, grid, m))
    field_vals = np.concatenate([getattr(m, args.field)[m.rock] for _, _, _, m in models])
    vmin, vmax = np.percentile(field_vals, 0.5), np.percentile(field_vals, 99.5)

    n = len(models)
    ncol = 2 if n > 3 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5 * ncol, 5.2 * nrow), squeeze=False, constrained_layout=True)
    for ax, (name, cfg, grid, m) in zip(axes.ravel(), models):
        f = getattr(m, args.field).astype(float)
        f[m.cavity] = np.nan
        extent = [grid.z0, grid.z0 + grid.dx * (grid.nz - 1), grid.x0, grid.x0 + grid.dx * (grid.nx - 1)]
        im = ax.imshow(f.T, origin="lower", extent=extent, aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax)
        src = shot_positions(cfg.acquisition, cfg.tunnel)
        rcv = receiver_positions(cfg.acquisition, cfg.tunnel)
        ax.plot(src[:, 0], src[:, 1], "r*", ms=5)
        ax.plot(rcv[:, 0], rcv[:, 1], "wv", mec="k", ms=7)
        sub = (f"{cfg.structure.family}, dx {grid.dx} m, {cfg.source.peak_freq:.0f} Hz"
               + (f", texture {cfg.rock.texture_std:.0%}/{cfg.rock.texture_corr_len} m" if cfg.rock.texture_std else "")
               + (f", EDZ {cfg.tunnel.edz_thickness} m x{cfg.tunnel.edz_factor}" if cfg.tunnel.enabled else ", no tunnel"))
        ax.set_title(f"{name}\n{sub}", fontsize=10)
        ax.set_xlabel("z_t along tunnel [m]")
        ax.set_ylabel("x_t across tunnel [m]")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=f"{args.field} [{'kg/m3' if args.field == 'rho' else 'm/s'}] (white = cavity)")
    fig.suptitle(f"{args.field} maps of all configurations (red stars = shots, triangles = receivers)", fontsize=13)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
