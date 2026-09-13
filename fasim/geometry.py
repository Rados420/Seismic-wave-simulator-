"""Plan-view coordinate frame, grid indexing and acquisition placement."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AcquisitionCfg, DomainCfg, TunnelCfg

# Staggered-grid offsets (in cells) of each Deepwave field relative to the
# integer (z, x) node where the model parameters live:
#   sigma_yy/xx, pressure   -> (0, 0)
#   v_y (along tunnel)      -> (+0.5, 0)
#   v_x (across tunnel)     -> (0, +0.5)
STAGGER = {"p": (0.0, 0.0), "y": (0.5, 0.0), "x": (0.0, 0.5)}


@dataclass
class Grid:
    z0: float           # coordinate of node 0 along dim 0
    x0: float           # coordinate of node 0 along dim 1
    dx: float
    nz: int
    nx: int

    @classmethod
    def from_domain(cls, d: DomainCfg) -> "Grid":
        nz = int(round((d.z_max - d.z_min) / d.dx)) + 1
        nx = int(round((d.x_max - d.x_min) / d.dx)) + 1
        return cls(z0=d.z_min, x0=d.x_min, dx=d.dx, nz=nz, nx=nx)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nz, self.nx)

    @property
    def z(self) -> np.ndarray:
        return self.z0 + self.dx * np.arange(self.nz)

    @property
    def x(self) -> np.ndarray:
        return self.x0 + self.dx * np.arange(self.nx)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.z, self.x, indexing="ij")

    def to_index(self, pos: np.ndarray, component: str = "p") -> tuple[np.ndarray, np.ndarray]:
        """Nearest grid index for physical positions (…, 2) = (z_t, x_t).

        Accounts for the staggered location of the requested field so the
        *staggered* node ends up closest to the requested point. Returns the
        integer indices and the actual physical positions of those nodes.
        """
        pos = np.asarray(pos, dtype=float)
        sz, sx = STAGGER[component]
        iz = np.round((pos[..., 0] - self.z0) / self.dx - sz).astype(int)
        ix = np.round((pos[..., 1] - self.x0) / self.dx - sx).astype(int)
        iz = np.clip(iz, 0, self.nz - 2)
        ix = np.clip(ix, 0, self.nx - 2)
        actual = np.stack([self.z0 + (iz + sz) * self.dx, self.x0 + (ix + sx) * self.dx], axis=-1)
        return np.stack([iz, ix], axis=-1), actual

    def to_frac_index(self, pos: np.ndarray) -> np.ndarray:
        """Fractional (non-staggered) index of physical positions, for interpolation."""
        pos = np.asarray(pos, dtype=float)
        return np.stack([(pos[..., 0] - self.z0) / self.dx, (pos[..., 1] - self.x0) / self.dx], axis=-1)


def wall_sign(side: str) -> float:
    if side == "right":
        return 1.0
    if side == "left":
        return -1.0
    raise ValueError(f"side must be left|right, got {side!r}")


def receiver_positions(acq: AcquisitionCfg, tun: TunnelCfg) -> np.ndarray:
    """(n_rcv, 2) positions [z_t, x_t] of the receivers (borehole bottoms)."""
    return np.array([[r.z, wall_sign(r.side) * (tun.half_width + r.embed)] for r in acq.receivers])


def receiver_names(acq: AcquisitionCfg) -> list[str]:
    return [r.name or f"RCV{i + 1}" for i, r in enumerate(acq.receivers)]


def shot_positions(acq: AcquisitionCfg, tun: TunnelCfg) -> np.ndarray:
    """(n_shots, 2) positions [z_t, x_t] of the shots (shot-hole bottoms)."""
    if acq.shot_z is not None:
        zs = np.asarray(acq.shot_z, dtype=float)
    else:
        zs = acq.shot_z0 + acq.shot_dz * np.arange(acq.n_shots)
    xs = np.full_like(zs, wall_sign(acq.shot_side) * (tun.half_width + acq.shot_embed))
    return np.stack([zs, xs], axis=-1)


def force_direction(acq: AcquisitionCfg, direction_deg: float) -> np.ndarray:
    """Unit vector (dz, dx) of the hammer force.

    0 deg = wall normal pointing into the rock (away from the tunnel),
    90 deg = along the tunnel towards the face (+z_t).
    """
    n = np.array([0.0, wall_sign(acq.shot_side)])          # into the rock
    a = np.array([1.0, 0.0])                                # along tunnel
    th = np.deg2rad(direction_deg)
    return np.cos(th) * n + np.sin(th) * a


def offsets(src: np.ndarray, rcv: np.ndarray) -> np.ndarray:
    """(n_src, n_rcv) in-plane Euclidean source-receiver distances."""
    return np.linalg.norm(src[:, None, :] - rcv[None, :, :], axis=-1)
