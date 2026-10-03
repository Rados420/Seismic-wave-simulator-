"""Figures for the generator: model views (no simulation needed) and variant comparisons."""
from __future__ import annotations

import json

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .build import LABELS, Domain, TunnelGeom

LABEL_COLORS = ["#c9c9c9", "#8fb3d9", "#d62728", "#ff9896", "#9467bd", "#bcbd22", "#000000", "#ffbb78",
                "#7f7f7f", "#ffffff"]


def _img(ax, f, ext, title, cmap="viridis", vmin=None, vmax=None, cbar=True, label=None):
    im = ax.imshow(f.T, origin="lower", extent=ext, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=9)
    if cbar:
        plt.colorbar(im, ax=ax, shrink=0.8, label=label)
    return im


def _label_img(ax, lab, ext, title):
    cmap = ListedColormap(LABEL_COLORS)
    ax.imshow(lab.T, origin="lower", extent=ext, aspect="equal", cmap=cmap, vmin=-0.5, vmax=len(LABELS) - 0.5,
              interpolation="nearest")
    ax.set_title(title, fontsize=9)
    present = sorted(set(np.unique(lab).tolist()))
    names = {v: k for k, v in LABELS.items()}
    handles = [plt.Rectangle((0, 0), 1, 1, fc=LABEL_COLORS[c], ec="k", lw=0.3) for c in present]
    ax.legend(handles, [names[c] for c in present], fontsize=6, loc="upper left", ncol=2)


def _vp_masked(m):
    f = m["vp"].astype(float).copy()
    f[m["vp"] <= 0] = np.nan
    return f


def plot_models(models: dict, dom: Domain, geom: TunnelGeom, rcv: np.ndarray, shots: np.ndarray,
                params: dict, path, title: str = ""):
    """2-D: full vp, labels, and a zoom on the wall per wall variant. 3-D: three orthogonal slices per variant."""
    walls = list(models)
    vals = np.concatenate([m["vp"][m["vp"] > 0].ravel()[::7] for m in models.values()])
    vmin, vmax = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
    if dom.dims == 2:
        ext = [*dom.extent(0), *dom.extent(1)]
        fig, axes = plt.subplots(3, len(walls), figsize=(9 * len(walls), 13), squeeze=False)
        z_s = (rcv[:, 0].min() - 3, shots[:6, 0].max() + 3)
        n_w = geom.n_c(params["tunnel"]["s_ref"]) + geom.R
        for j, w in enumerate(walls):
            m = models[w]
            _img(axes[0, j], _vp_masked(m), ext, f"{w}: vp", vmin=vmin, vmax=vmax, label="m/s")
            _label_img(axes[1, j], m["labels"], ext, f"{w}: labels")
            _img(axes[2, j], _vp_masked(m), ext, f"{w}: vp, wall zoom", vmin=vmin, vmax=vmax, label="m/s")
            axes[2, j].set_xlim(*z_s)
            axes[2, j].set_ylim(n_w - 2.5, n_w + 4)
            for ax in axes[:, j]:
                ax.plot(shots[:, 0], shots[:, 1], "r*", ms=5)
                ax.plot(rcv[:, 0], rcv[:, 1], "wv", mec="k", ms=6)
                ax.set_xlabel("s along tunnel [m]")
                ax.set_ylabel("n into rock [m]")
    else:
        fig, axes = plt.subplots(3, len(walls), figsize=(9 * len(walls), 15), squeeze=False)
        iv = int(np.clip(round((0.0 - dom.origin[2]) / dom.dx), 0, dom.shape[2] - 1))
        s_mid = float(np.median(shots[:, 0]))
        i_s = int(np.clip(round((s_mid - dom.origin[0]) / dom.dx), 0, dom.shape[0] - 1))
        n_r = float(np.median(rcv[:, 1]))
        i_n = int(np.clip(round((n_r - dom.origin[1]) / dom.dx), 0, dom.shape[1] - 1))
        for j, w in enumerate(walls):
            f = _vp_masked(models[w])
            _img(axes[0, j], f[:, :, iv], [*dom.extent(0), *dom.extent(1)],
                 f"{w}: plan view at v = {dom.origin[2] + iv * dom.dx:.1f} m", vmin=vmin, vmax=vmax, label="m/s")
            axes[0, j].plot(shots[:, 0], shots[:, 1], "r*", ms=5)
            axes[0, j].plot(rcv[:, 0], rcv[:, 1], "wv", mec="k", ms=6)
            axes[0, j].set_xlabel("s [m]"); axes[0, j].set_ylabel("n [m]")
            _img(axes[1, j], f[i_s, :, :], [*dom.extent(1), *dom.extent(2)],
                 f"{w}: cross-section at s = {s_mid:.1f} m", vmin=vmin, vmax=vmax, label="m/s")
            axes[1, j].set_xlabel("n [m]"); axes[1, j].set_ylabel("v [m]")
            _img(axes[2, j], f[:, i_n, :], [*dom.extent(0), *dom.extent(2)],
                 f"{w}: along-wall section at n = {n_r:.1f} m (receiver depth)", vmin=vmin, vmax=vmax, label="m/s")
            axes[2, j].plot(shots[:, 0], shots[:, 2], "r*", ms=5)
            axes[2, j].plot(rcv[:, 0], rcv[:, 2], "wv", mec="k", ms=6)
            axes[2, j].set_xlabel("s [m]"); axes[2, j].set_ylabel("v [m]")
    fig.suptitle(title, fontsize=10, family="monospace", ha="left", x=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _wiggles(ax, d, t, eik, title, gain=0.6):
    """Traces as rows (first shot on top), time on x; d already normalised."""
    n = d.shape[0]
    for i in range(n):
        y0 = n - 1 - i
        tr = np.clip(d[i], -1.5, 1.5) * gain
        ax.plot(t * 1e3, y0 + tr, "k", lw=0.4)
        ax.fill_between(t * 1e3, y0, y0 + tr, where=tr > 0, color="k", lw=0)
    if eik is not None:
        ys = n - 1 - np.arange(n)
        ax.vlines(eik * 1e3, ys - 0.45, ys + 0.45, colors="tab:blue", lw=1.2)
    ax.set_ylim(-1, n)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time rel. wavelet peak [ms]")


def plot_compare(path_h5, rcv: int = 0, comp: str = "n", out=None, tmax_ms: float | None = None):
    """Receiver gather for every (wall, source) variant + wall difference per source.

    Each trace is normalised by the max of the same trace in the reference wall variant
    (the last listed wall), so amplitude changes between walls stay visible.
    """
    with h5py.File(path_h5, "r") as h:
        comps = json.loads(h.attrs["components"])
        meta = json.loads(h.attrs["meta_json"])
        walls, sources = meta["enumerate"]["walls"], meta["enumerate"]["sources"]
        ci = comps.index(comp)
        g = h["geometry"]
        offset = g["offset"][:, rcv]
        D = {(w, s): h[f"wall_{w}/{s}/data"][:, rcv, ci, :].astype(float) for w in walls for s in sources}
        T = {s: h[f"wall_{walls[0]}/{s}/t"][()] for s in sources}
        E = {w: (h[f"wall_{w}/eikonal"][:, rcv] if "eikonal" in h[f"wall_{w}"] else None) for w in walls}
        seed = int(h.attrs["seed"])
    ref_w = walls[-1]
    nrow = len(walls) + 1
    fig, axes = plt.subplots(nrow, len(sources), figsize=(7 * len(sources), 3.2 * nrow + 0.4 * len(offset) / 4),
                             squeeze=False, sharey=True)
    for j, s in enumerate(sources):
        ref = np.abs(D[(ref_w, s)]).max(axis=1, keepdims=True) + 1e-30
        tm = tmax_ms if tmax_ms else min(T[s][-1] * 1e3, (np.nanmax(E[ref_w]) * 1e3 + 10) if E[ref_w] is not None else 30)
        for i, w in enumerate(walls):
            _wiggles(axes[i, j], D[(w, s)] / ref, T[s], E[w], f"wall {w}, {s}  (comp {comp})")
            axes[i, j].set_xlim(T[s][0] * 1e3, tm)
        if len(walls) >= 2:
            diff = (D[(walls[0], s)] - D[(walls[1], s)]) / ref
            _wiggles(axes[-1, j], diff, T[s], None, f"{walls[0]} - {walls[1]}, {s} (same scale)")
            axes[-1, j].set_xlim(T[s][0] * 1e3, tm)
    for ax in axes[:, 0]:
        ax.set_yticks(len(offset) - 1 - np.arange(len(offset)))
        ax.set_yticklabels([f"{o:.1f}" for o in offset], fontsize=6)
        ax.set_ylabel("offset [m]")
    fig.suptitle(f"draw {seed}, receiver {rcv}, component {comp}; blue = eikonal", fontsize=10)
    fig.tight_layout()
    out = out or str(path_h5).replace(".h5", f"_compare_rcv{rcv}_{comp}.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out
