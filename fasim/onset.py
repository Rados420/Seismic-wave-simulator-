"""Observable-onset pickers for clean (noise-free) synthetic traces.

All pickers share the signature  pick(trace, dt, i0=0, **kw) -> onset time [s]
measured on the sample axis (sample k at time k*dt), sub-sample interpolated,
NaN when nothing is found. `i0` is the first sample of the search window.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from .config import LabelCfg


def _crossing(x: np.ndarray, thr: float, i0: int, dt: float) -> float:
    if not np.isfinite(thr):
        return np.nan
    idx = np.nonzero(x[i0:] > thr)[0]
    if idx.size == 0:
        return np.nan
    k = idx[0] + i0
    if idx[0] == 0:
        # already above threshold at the start of the search window: no crossing to interpolate
        return k * dt
    a, b = x[k - 1], x[k]
    frac = float(np.clip((thr - a) / (b - a), 0.0, 1.0)) if b != a else 0.0
    return (k - 1 + frac) * dt


def _reference(x, i0, ref_amp, ref_window, floor=0.0):
    """Reference amplitude for relative thresholds (see LabelCfg.threshold_ref).

    Returns NaN (-> no pick) when the reference is below `floor`, i.e. nothing
    observable arrives (shadowed receiver, numerical background only).
    """
    if ref_amp is not None:                      # gather-level reference supplied by the caller
        ref = ref_amp
    elif ref_window is not None:                 # window after the search start
        ref = x[i0:i0 + max(ref_window, 1)].max()
    else:
        ref = x[i0:].max()                       # whole (remaining) trace
    return ref if ref > floor else np.nan


def envelope_threshold(trace, dt, i0=0, rel_threshold=0.02, ref_amp=None, ref_window=None, floor=0.0, **_):
    env = np.abs(hilbert(trace))
    return _crossing(env, rel_threshold * _reference(env, i0, ref_amp, ref_window, floor), i0, dt)


def amplitude_threshold(trace, dt, i0=0, rel_threshold=0.02, ref_amp=None, ref_window=None, floor=0.0, **_):
    a = np.abs(trace)
    return _crossing(a, rel_threshold * _reference(a, i0, ref_amp, ref_window, floor), i0, dt)


def sta_lta(trace, dt, i0=0, sta_ms=0.1, lta_ms=1.0, trigger=5.0, **_):
    """Classic STA/LTA on squared amplitude, causal moving averages."""
    x = trace.astype(np.float64) ** 2
    ns, nl = max(int(sta_ms * 1e-3 / dt), 1), max(int(lta_ms * 1e-3 / dt), 2)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    k = np.arange(len(x)) + 1
    sta = (cs[k] - cs[np.maximum(k - ns, 0)]) / np.minimum(k, ns)
    lta = (cs[k] - cs[np.maximum(k - nl, 0)]) / np.minimum(k, nl)
    ratio = sta / (lta + 1e-30 * x.max())
    ratio[:nl] = 0.0
    return _crossing(ratio, trigger, max(i0, nl), dt)


def aic(trace, dt, i0=0, i1=None, **_):
    """Akaike information criterion picker on the window [i0, i1)."""
    x = trace[i0:i1].astype(np.float64)
    n = len(x)
    if n < 4:
        return np.nan
    k = np.arange(1, n - 1)
    v1 = np.array([np.var(x[:i]) for i in k]) + 1e-30 * x.var()
    v2 = np.array([np.var(x[i:]) for i in k]) + 1e-30 * x.var()
    a = k * np.log(v1) + (n - k) * np.log(v2)
    return (i0 + k[np.argmin(a)]) * dt


PICKERS = {
    "envelope_threshold": envelope_threshold,
    "amplitude_threshold": amplitude_threshold,
    "sta_lta": sta_lta,
    "aic": aic,
}


def picker_kwargs(lc: LabelCfg) -> dict:
    return dict(rel_threshold=lc.rel_threshold, sta_ms=lc.sta_ms, lta_ms=lc.lta_ms, trigger=lc.trigger)


def _picker_only(kw: dict) -> dict:
    return {k: v for k, v in kw.items() if k not in ("threshold_ref", "ref_window_ms")}


def component_trace(data_tr: np.ndarray, component: str) -> np.ndarray:
    """data_tr [comp, nt] -> 1-D trace for a component name (y | x | p | vec)."""
    if component == "vec":
        return np.hypot(data_tr[0], data_tr[1])
    return data_tr[("y", "x", "p").index(component)]


def pick_gather(data: np.ndarray, t: np.ndarray, lc: LabelCfg, eikonal: np.ndarray | None = None,
                w_onset: float = 0.0, picker: str | None = None, verbose: bool = True, **overrides) -> np.ndarray:
    """Pick onsets for data [shot, rcv, comp, nt] -> [shot, rcv, comp+1] (last = configured component).

    Times are returned on the same axis as `t` (0 = wavelet peak). If `eikonal`
    [shot, rcv, comp] is given, the search starts `search_before_ms` before the
    expected first energy, eikonal + w_onset (w_onset = picker onset of the
    source wavelet relative to its peak, a negative number).
    """
    fn = PICKERS[picker or lc.picker]
    kw = picker_kwargs(lc) | overrides
    ref_mode = kw.pop("threshold_ref", lc.threshold_ref)
    ref_window = int(round(kw.pop("ref_window_ms", lc.ref_window_ms) * 1e-3 / float(t[1] - t[0])))
    dt = float(t[1] - t[0])
    n_shot, n_rcv, n_comp, nt = data.shape
    out = np.full((n_shot, n_rcv, n_comp + 1), np.nan)
    comps = list(("y", "x", "p")[:n_comp]) + [lc.component]
    at_start = 0
    # per-component observability floor (pressure and velocity have very different units): a reference
    # amplitude below -80 dB re. the strongest trace of the gather is not a real arrival -> NaN label
    floor = {c: 1e-4 * max(np.abs(component_trace(data[k, q], c)).max()
                           for k in range(n_shot) for q in range(n_rcv)) for c in set(comps)}
    for s in range(n_shot):
        for r in range(n_rcv):
            for ci, c in enumerate(comps):
                tr = component_trace(data[s, r], c)
                if np.abs(tr).max() < floor[c]:
                    continue
                i0 = 0
                if eikonal is not None and lc.search_before_ms is not None:
                    ce = min(ci, n_comp - 1) if c != "vec" else 1
                    t_start = eikonal[s, r, ce] + w_onset - lc.search_before_ms * 1e-3
                    i0 = int(np.clip(np.searchsorted(t, t_start), 0, nt - 2))
                ref_amp = None
                if ref_mode == "gather":
                    g = np.stack([component_trace(data[k, r], c) for k in range(n_shot)])
                    ref_amp = np.abs(hilbert(g, axis=-1)).max() if fn is envelope_threshold else np.abs(g).max()
                tp = fn(tr, dt, i0=i0, ref_amp=ref_amp, ref_window=ref_window if ref_mode == "window" else None,
                        floor=floor[c], **kw)
                out[s, r, ci] = tp + t[0] if np.isfinite(tp) else np.nan
                if np.isfinite(tp) and abs(tp - i0 * dt) < 0.5 * dt:
                    at_start += 1
    if verbose and at_start:
        print(f"  WARNING: {at_start} picks sit at the search-window start; increase labels.search_before_ms")
    return out


def wavelet_onset(wavelet: np.ndarray, dt: float, peak_time: float, lc: LabelCfg,
                  picker: str | None = None, **overrides) -> float:
    """Onset of the source wavelet itself, relative to its peak (negative number)."""
    fn = PICKERS[picker or lc.picker]
    tp = fn(np.asarray(wavelet), dt, i0=0, **_picker_only(picker_kwargs(lc) | overrides))
    return tp - peak_time


def emergence_delay(onset: np.ndarray, eikonal: np.ndarray, w_onset: float) -> np.ndarray:
    """onset - (eikonal + wavelet onset): 0 for a clean straight-ray arrival."""
    return onset - (eikonal + w_onset)
