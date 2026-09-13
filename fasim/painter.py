"""Paint vp / vs / rho grids from implicit geometry (boolean masks on coordinates)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import ExperimentCfg, gardner_rho
from .geometry import Grid


@dataclass
class Model:
    vp: np.ndarray            # [m/s], 0 inside the cavity
    vs: np.ndarray
    rho: np.ndarray           # [kg/m^3], 0 inside the cavity
    cavity: np.ndarray        # bool mask
    edz: np.ndarray           # bool mask
    structure: np.ndarray     # bool mask of the structural anomaly (fault / layer)

    @property
    def rock(self) -> np.ndarray:
        return ~self.cavity

    def vp_intact_max(self) -> float:
        return float(self.vp.max())


def _band_distance(Z, X, z_cross: float, strike_deg: float) -> np.ndarray:
    """Signed distance from a straight line crossing x_t=0 at z_cross.

    strike_deg is the angle between the line and the tunnel axis (dim 0):
    90 -> line perpendicular to the tunnel (a plane ahead of the face).
    """
    th = np.deg2rad(strike_deg)
    # line direction (dz, dx) = (cos th, sin th); normal = (-sin th, cos th)
    nz, nx = -np.sin(th), np.cos(th)
    # choose the sign so that "ahead" (increasing z_t at x_t = 0) is positive
    if nz < 0:
        nz, nx = -nz, -nx
    return (Z - z_cross) * nz + X * nx


def _cavity_masks(cfg: ExperimentCfg, Z, X):
    tun = cfg.tunnel
    if not tun.enabled:
        return np.zeros(Z.shape, bool), np.zeros(Z.shape, bool)
    z_start = tun.z_start if tun.z_start is not None else Z.min() - 1e3
    cavity = (np.abs(X) < tun.half_width) & (Z >= z_start) & (Z <= tun.z_face)
    # distance from the cavity rectangle (0 inside)
    dz = np.maximum(np.maximum(z_start - Z, Z - tun.z_face), 0.0)
    dxx = np.maximum(np.abs(X) - tun.half_width, 0.0)
    dist = np.hypot(dz, dxx)
    edz = (~cavity) & (dist < tun.edz_thickness) if tun.edz_thickness > 0 else np.zeros(Z.shape, bool)
    return cavity, edz


def paint(cfg: ExperimentCfg, grid: Grid, rng: np.random.Generator | None = None) -> Model:
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    Z, X = grid.meshgrid()
    rock, st = cfg.rock, cfg.structure

    vp = np.full(grid.shape, rock.vp, dtype=np.float64)
    vp_vs = np.full(grid.shape, rock.vp_vs, dtype=np.float64)

    # --- large-scale structure -------------------------------------------- #
    if st.family == "homogeneous":
        structure = np.zeros(grid.shape, bool)
    elif st.family == "fault_zone":
        d = _band_distance(Z, X, st.fault_z, st.fault_strike_deg)
        structure = np.abs(d) < 0.5 * st.fault_width
        vp[structure] = st.fault_vp
        vp_vs[structure] = st.fault_vp_vs
    elif st.family == "layered_dip":
        d = _band_distance(Z, X, st.layer_z, st.layer_strike_deg)
        structure = d > 0
        vp[structure] = st.layer_vp
        vp_vs[structure] = st.layer_vp_vs
    else:
        raise ValueError(f"unknown structure family {st.family!r}")

    # --- metre-scale texture (multiplicative, correlated) ------------------ #
    if rock.texture_std > 0:
        sigma_cells = max(rock.texture_corr_len / grid.dx, 0.5)
        noise = gaussian_filter(rng.standard_normal(grid.shape), sigma_cells)
        noise /= noise.std() + 1e-12
        factor = 1.0 + rock.texture_std * noise
        vp *= factor

    vs = vp / vp_vs
    rho = gardner_rho(vp) if rock.rho is None else np.full(grid.shape, float(rock.rho))

    # --- tunnel: EDZ then cavity ------------------------------------------- #
    cavity, edz = _cavity_masks(cfg, Z, X)
    vp[edz] *= cfg.tunnel.edz_factor
    vs[edz] *= cfg.tunnel.edz_factor
    rho[edz] *= cfg.tunnel.edz_rho_factor
    vp[cavity] = 0.0
    vs[cavity] = 0.0
    rho[cavity] = 0.0        # vp=vs=rho=0 -> lamb=mu=buoyancy=0 -> Deepwave vacuum (free surface)

    return Model(vp=vp.astype(np.float32), vs=vs.astype(np.float32), rho=rho.astype(np.float32),
                 cavity=cavity, edz=edz, structure=structure)


def min_wavelength(model: Model, fmax: float) -> float:
    vs_rock = model.vs[model.rock]
    return float(vs_rock[vs_rock > 0].min() / fmax)
