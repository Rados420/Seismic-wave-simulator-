"""Correlated random fields (FFT spectral synthesis)."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft


def von_karman(shape, dx: float, corr_len: float, hurst: float, seed: int,
               normal=None, ratio: float = 1.0, pad=None) -> np.ndarray:
    """Zero-mean, unit-std field with a von Karman spectrum, float32.

    normal / ratio: anisotropy, the correlation length along `normal` is corr_len / ratio
    (fabric planes). For a 2-D grid only the first two components of a 3-D normal are used,
    which is the in-plane projection. pad: per axis, whether to pad against FFT wrap-around
    (False for periodic axes).
    """
    rng = np.random.default_rng(seed)
    ndim = len(shape)
    pad = [True] * ndim if pad is None else pad
    extra = int(np.ceil(corr_len / dx))
    pshape = tuple(n + (min(extra, n) if p else 0) for n, p in zip(shape, pad))

    ks = [sfft.fftfreq(n, d=dx) * 2 * np.pi for n in pshape[:-1]]
    ks.append(sfft.rfftfreq(pshape[-1], d=dx) * 2 * np.pi)
    K = np.meshgrid(*[k.astype(np.float32) for k in ks], indexing="ij", sparse=True)
    k2 = sum(k * k for k in K)
    if normal is not None and ratio != 1.0:
        nv = np.asarray(normal, float)[:ndim]
        kn = sum(k * float(c) for k, c in zip(K, nv))
        keff2 = corr_len ** 2 * (k2 - kn * kn) + (corr_len / ratio) ** 2 * kn * kn
    else:
        keff2 = corr_len ** 2 * k2
    amp = (1.0 + keff2) ** (-(hurst + ndim / 2.0) / 2.0)

    white = rng.standard_normal(pshape, dtype=np.float32)
    f = sfft.irfftn(sfft.rfftn(white, workers=-1) * amp, s=pshape, workers=-1)
    f = f[tuple(slice(0, n) for n in shape)]
    f -= f.mean()
    f /= f.std() + 1e-12
    return f.astype(np.float32)
