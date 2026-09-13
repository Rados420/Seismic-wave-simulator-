"""Kinematic reference label: fast-marching traveltimes on the same vp grid, cavity masked."""
from __future__ import annotations

import numpy as np
import skfmm
from scipy.ndimage import map_coordinates

from .geometry import Grid
from .painter import Model


def traveltime_field(grid: Grid, model: Model, src: np.ndarray, order: int = 2) -> np.ndarray:
    """First-arrival traveltime [s] from a point source at src = (z_t, x_t); NaN in the cavity."""
    Z, X = grid.meshgrid()
    r0 = grid.dx                       # radius of the initial front
    phi = np.hypot(Z - src[0], X - src[1]) - r0
    speed = model.vp.astype(np.float64)
    if model.cavity.any():
        phi = np.ma.MaskedArray(phi, mask=model.cavity)
        speed = np.ma.MaskedArray(speed, mask=model.cavity)
    tt = skfmm.travel_time(phi, speed, dx=grid.dx, order=order)
    tt = np.ma.filled(tt, np.nan) if np.ma.isMaskedArray(tt) else np.asarray(tt)
    # correct for the finite initial radius
    iz, ix = grid.to_index(src[None, :], "p")[0][0]
    v_src = model.vp[iz, ix]
    if v_src <= 0:
        raise ValueError("source lies inside the cavity")
    tt = tt + r0 / v_src
    # inside the initial circle travel_time is ~0 -> replace by straight-ray value
    inside = phi < 0 if not np.ma.isMaskedArray(phi) else np.ma.filled(phi < 0, False)
    tt[inside] = np.hypot(Z[inside] - src[0], X[inside] - src[1]) / v_src
    return tt


def sample(field: np.ndarray, grid: Grid, pos: np.ndarray) -> np.ndarray:
    """Bilinear sample of a grid field at physical positions (..., 2)."""
    fi = grid.to_frac_index(pos)
    flat = fi.reshape(-1, 2).T
    f = np.nan_to_num(field, nan=np.nanmax(field))   # cavity NaNs -> large, only matters right at the wall
    vals = map_coordinates(f, flat, order=1, mode="nearest")
    return vals.reshape(pos.shape[:-1])


def eikonal_labels(grid: Grid, model: Model, src_pos: np.ndarray, rcv_actual: np.ndarray,
                   verbose: bool = False) -> np.ndarray:
    """Eikonal traveltime [s] for every (shot, rcv, comp) using the actual staggered receiver positions."""
    n_shots = src_pos.shape[0]
    out = np.zeros((n_shots,) + rcv_actual.shape[:-1])
    for s in range(n_shots):
        tt = traveltime_field(grid, model, src_pos[s])
        out[s] = sample(tt, grid, rcv_actual)
        if verbose and s % 8 == 0:
            print(f"  eikonal shot {s}", flush=True)
    return out
