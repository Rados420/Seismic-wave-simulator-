"""Post-simulation amplitude physics that the lossless 2-D elastic solver does not provide.

Two effects, both applied per trace on the time axis t (0 = source wavelet peak):

* constant-Q intrinsic attenuation: every frequency component is damped by
  exp(-pi f t / Q) as a function of propagation time t (Kolsky, no dispersion).
  Implemented as an exact time-variant filter: y(t) = Re sum_f X(f) exp(-pi f t/Q) e^{2 pi i f t}.
* geometric spreading correction: multiply by (t / t_ref)^(-p). 2-D waves spread
  as 1/sqrt(r) (body) and not at all (surface); 3-D as 1/r and 1/sqrt(r). p = 0.5
  converts both classes by one order, which is the usual 2-D -> 3-D amplitude proxy.
"""
from __future__ import annotations

import numpy as np


def constant_q(data: np.ndarray, t: np.ndarray, q: float, f_max: float | None = None) -> np.ndarray:
    """data [..., nt] -> attenuated copy. t [nt] in seconds, negative times treated as 0."""
    if q is None or not np.isfinite(q) or q <= 0:
        return data
    nt = t.size
    dt = float(t[1] - t[0])
    f = np.fft.rfftfreq(nt, dt)                        # [nf]
    if f_max is not None:
        f = f[f <= f_max]
    tt = np.clip(t, 0.0, None)                         # [nt]
    # kernel K[t, f] = exp(-pi f t / Q) * exp(2 pi i f t) / nt, weight 2 for f>0 (real signal)
    w = np.full(f.size, 2.0 / nt)
    w[0] = 1.0 / nt
    if nt % 2 == 0 and f_max is None:
        w[-1] = 1.0 / nt
    # the FFT phase is relative to sample 0, so the synthesis exponent uses sample times k*dt,
    # while the damping uses the physical propagation time tt
    k = np.arange(nt) * dt
    kern = (w[None, :] * np.exp(-np.pi * f[None, :] * tt[:, None] / q)
            * np.exp(2j * np.pi * f[None, :] * k[:, None]))
    X = np.fft.rfft(data, axis=-1)[..., : f.size]      # [..., nf]
    y = np.real(X @ kern.T)                            # [..., nt]
    return y.astype(data.dtype)


def spreading(data: np.ndarray, t: np.ndarray, exponent: float, t_ref: float = 1e-3) -> np.ndarray:
    """Multiply by (t / t_ref)^(-exponent) for t > t_ref (unchanged before)."""
    if not exponent:
        return data
    g = np.ones_like(t)
    m = t > t_ref
    g[m] = (t[m] / t_ref) ** (-exponent)
    return (data * g).astype(data.dtype)


def apply(data: np.ndarray, t: np.ndarray, q: float | None, spreading_exponent: float) -> np.ndarray:
    out = constant_q(data, t, q) if q else data
    return spreading(out, t, spreading_exponent)
