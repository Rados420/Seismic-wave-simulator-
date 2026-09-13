"""Plots: model map, wiggle gathers with labels, delay vs offset, wavefield snapshots."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .geometry import Grid
from .painter import Model


def plot_model(model: Model, grid: Grid, src_pos=None, rcv_pos=None, rcv_names=None, ax=None,
               field: str = "vp", title: str | None = None):
    ax = ax or plt.gca()
    f = getattr(model, field).astype(float).copy()
    f[model.cavity] = np.nan
    extent = [grid.z0, grid.z0 + grid.dx * (grid.nz - 1), grid.x0, grid.x0 + grid.dx * (grid.nx - 1)]
    im = ax.imshow(f.T, origin="lower", extent=extent, aspect="equal", cmap="viridis")
    plt.colorbar(im, ax=ax, label=f"{field} [m/s]" if field != "rho" else "rho [kg/m3]", shrink=0.7)
    if src_pos is not None:
        ax.plot(src_pos[:, 0], src_pos[:, 1], "r*", ms=6, label="shots")
    if rcv_pos is not None:
        ax.plot(rcv_pos[:, 0], rcv_pos[:, 1], "wv", mec="k", ms=8, label="receivers")
        if rcv_names:
            for (z, x), n in zip(rcv_pos, rcv_names):
                ax.annotate(n, (z, x), textcoords="offset points", xytext=(4, 4), color="w", fontsize=8)
    ax.set_xlabel("z_t along tunnel [m]")
    ax.set_ylabel("x_t across tunnel [m]")
    ax.set_title(title or f"{field} (white = cavity)")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def _normalise(d, mode):
    if mode == "trace":
        return d / (np.abs(d).max(axis=1, keepdims=True) + 1e-30)
    if mode == "global":
        return d / (np.abs(d).max() + 1e-30)
    return d


def plot_gather(data, t, labels=None, picks=None, ax=None, normalise="trace", gain=0.8, fill=True,
                tmin=None, tmax=None, title=None, ylabel="trace", time_unit="ms"):
    """Wiggle plot, time on the x-axis, one row per trace (first trace on top) — same look as tsp.plot.

    data [ntr, nt]; picks = {name: (times[ntr], color)}; times on the same axis as t.
    """
    ax = ax or plt.gca()
    scale = 1e3 if time_unit == "ms" else 1.0
    tt = t * scale
    d = _normalise(np.asarray(data, float), normalise) * gain
    ntr = d.shape[0]
    for i in range(ntr):
        y0 = ntr - 1 - i
        ax.plot(tt, y0 + d[i], "k", lw=0.5)
        if fill:
            ax.fill_between(tt, y0, y0 + d[i], where=d[i] > 0, color="k", lw=0)
    if picks:
        for name, (times, color) in picks.items():
            times = np.asarray(times, float) * scale
            ys = ntr - 1 - np.arange(ntr)
            ok = np.isfinite(times)
            ax.vlines(times[ok], ys[ok] - 0.45, ys[ok] + 0.45, colors=color, lw=1.5, label=name)
        ax.legend(loc="upper right", fontsize=8)
    if labels is not None:
        ax.set_yticks(ntr - 1 - np.arange(ntr))
        ax.set_yticklabels([f"{l:.1f}" if isinstance(l, (float, np.floating)) else str(l) for l in labels], fontsize=7)
    ax.set_xlim(tmin * scale if tmin is not None else tt[0], tmax * scale if tmax is not None else tt[-1])
    ax.set_ylim(-1, ntr)
    ax.set_xlabel(f"time rel. wavelet peak [{time_unit}]")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax


def plot_delay_vs_offset(offset, delay, rcv_names=None, ax=None, unit="ms", title="emergence delay"):
    """offset [shot, rcv], delay [shot, rcv] (already reduced to one component)."""
    ax = ax or plt.gca()
    scale = 1e3 if unit == "ms" else 1.0
    for r in range(offset.shape[1]):
        ax.plot(offset[:, r], delay[:, r] * scale, "o-", ms=4, lw=0.8,
                label=rcv_names[r] if rcv_names else f"rcv {r}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("offset [m]")
    ax.set_ylabel(f"onset - (eikonal + wavelet onset) [{unit}]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_snapshot(field, grid: Grid, model: Model, src_pos=None, rcv_pos=None, ax=None, title=None, clip=0.98):
    ax = ax or plt.gca()
    extent = [grid.z0, grid.z0 + grid.dx * (grid.nz - 1), grid.x0, grid.x0 + grid.dx * (grid.nx - 1)]
    v = np.quantile(np.abs(field), clip) + 1e-30
    ax.imshow(field.T, origin="lower", extent=extent, cmap="seismic", vmin=-v, vmax=v, aspect="equal")
    ax.contour(grid.z, grid.x, model.cavity.T.astype(float), levels=[0.5], colors="k", linewidths=0.8)
    if model.edz.any():
        ax.contour(grid.z, grid.x, (model.cavity | model.edz).T.astype(float), levels=[0.5], colors="gray", linewidths=0.5)
    if src_pos is not None:
        ax.plot(src_pos[:, 0], src_pos[:, 1], "r*", ms=8)
    if rcv_pos is not None:
        ax.plot(rcv_pos[:, 0], rcv_pos[:, 1], "gv", mec="k", ms=7)
    ax.set_xlabel("z_t [m]")
    ax.set_ylabel("x_t [m]")
    if title:
        ax.set_title(title)
    return ax


def plot_decay(data, t, offset, rcv_names=None, ax=None, comp=1, rcv=0, shots=None, win_ms=1.0, title=None):
    """Log RMS envelope (moving RMS in win_ms windows) vs time for a few traces of one receiver.

    Shows how fast energy dies out; real hard-rock records typically fall 40-60 dB within 30-50 ms.
    """
    ax = ax or plt.gca()
    dt = float(t[1] - t[0])
    n = max(int(win_ms * 1e-3 / dt), 1)
    kern = np.ones(n) / n
    nshot = data.shape[0]
    shots = shots if shots is not None else np.unique(np.linspace(0, nshot - 1, min(6, nshot)).astype(int))
    ref = None
    for s in shots:
        tr = data[s, rcv, comp].astype(float)
        rms = np.sqrt(np.convolve(tr ** 2, kern, mode="same"))
        ref = ref if ref is not None else rms.max()
        db = 20 * np.log10(rms / ref + 1e-30)
        ax.plot(t * 1e3, db, lw=0.9, label=f"offset {offset[s, rcv]:.1f} m")
    ax.set_ylim(-100, 5)
    ax.set_xlabel("time rel. wavelet peak [ms]")
    ax.set_ylabel("moving RMS [dB re. loudest trace]")
    ax.set_title(title or f"amplitude decay, {rcv_names[rcv] if rcv_names else rcv}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax
