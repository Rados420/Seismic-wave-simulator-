"""Turn a sampled draw into vp / vs / rho grids for one wall variant, in 2-D or 3-D.

Axes: dim 0 = s (along the tunnel), dim 1 = n (into the rock, normal to the receiver wall),
dim 2 = v (vertical, 3-D only). The box is the one-wall domain: the rock behind the receiver
wall plus `cavity_margin` of tunnel interior.

2-D projection: the geology (structures, voids) is the exact v = 0 slice of the 3-D geology;
the tunnel's vertical curvature and all heights (receivers, shots) are ignored. Texture fields
are generated directly on the 2-D grid (same statistics, not a slice of the 3-D realisation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from ..config import gardner_rho
from .fields import von_karman

# label codes stored with every model
LABELS = {"host": 0, "interface": 1, "fault_core": 2, "fault_halo": 3, "dyke": 4,
          "karst_clay": 5, "karst_void": 6, "edz": 7, "concrete": 8, "cavity": 9}
STRUCT_ORDER = ("interface", "gradient", "fault_zone", "dyke", "karst")


# --------------------------------------------------------------------------- #
class TunnelGeom:
    """Circular tunnel whose axis follows a parabola: n_c(s), v_c(s) (small-curvature approximation)."""

    def __init__(self, tp: dict, dims: int):
        self.R = float(tp["radius"])
        self.kh = float(tp["curvature_horizontal"])
        self.kv = float(tp["curvature_vertical"]) if dims == 3 else 0.0
        self.s_ref = float(tp["s_ref"])
        self.face_s = float(tp["face_s"])
        self.dims = dims

    def n_c(self, s):
        return -self.R + 0.5 * self.kh * (np.asarray(s) - self.s_ref) ** 2

    def v_c(self, s):
        return 0.5 * self.kv * (np.asarray(s) - self.s_ref) ** 2

    def theta(self, height):
        if self.dims == 2:
            return 0.0
        return float(np.arcsin(np.clip(height / self.R, -0.95, 0.95)))

    def wall_point(self, s: float, embed: float, height: float) -> np.ndarray:
        """Point `embed` metres beyond the nominal wall, radially, at `height` (ignored in 2-D)."""
        th = self.theta(height)
        rr = self.R + embed
        n = float(self.n_c(s)) + rr * np.cos(th)
        if self.dims == 2:
            return np.array([s, n])
        return np.array([s, n, float(self.v_c(s)) + rr * np.sin(th)])

    def outward(self, height: float) -> np.ndarray:
        th = self.theta(height)
        return np.array([0.0, np.cos(th)]) if self.dims == 2 else np.array([0.0, np.cos(th), np.sin(th)])


class Roughness:
    """Excavation-surface perturbation r_rock(s, theta) = R + amp * (field + bias), periodic in theta."""

    def __init__(self, rp: dict, R: float, s_lo: float, s_hi: float, step: float):
        self.amp, self.bias = float(rp["amplitude"]), float(rp["overbreak_bias"])
        self.s0, self.step, self.R = s_lo, step, R
        n_s = int(np.ceil((s_hi - s_lo) / step)) + 2
        n_a = max(int(round(2 * np.pi * R / step)), 8)
        self.n_a = n_a
        self.field = von_karman((n_s, n_a), step, float(rp["corr_length"]), 0.5, int(rp["field_seed"]),
                                pad=[True, False])

    def __call__(self, s, theta):
        s, theta = np.broadcast_arrays(s, theta)
        i_s = (s - self.s0) / self.step
        i_a = np.mod(theta * self.R / self.step, self.n_a)
        f = map_coordinates(self.field, [i_s.ravel(), i_a.ravel()], order=1, mode="grid-wrap")
        return self.amp * (f.reshape(s.shape) + self.bias)

    @property
    def min(self) -> float:
        return float(self.amp * (self.field.min() + self.bias))


@dataclass
class Domain:
    origin: np.ndarray
    dx: float
    shape: tuple
    dims: int
    hi: np.ndarray | None = None          # requested upper bounds (independent of dx)
    field_dx: float | None = None         # spacing on which random fields are generated (None = dx)

    def field(self, corr_len: float, hurst: float, seed: int, normal=None, ratio: float = 1.0) -> np.ndarray:
        """von Karman field on this grid. With field_dx set, the field is generated on a fixed grid of
        spacing field_dx over the dx-independent bounds and sampled here, so the same draw gives the
        same rock at every dx (needed for grid-convergence tests)."""
        if not self.field_dx or abs(self.field_dx - self.dx) < 1e-9 or self.hi is None:
            return von_karman(self.shape, self.dx, corr_len, hurst, seed, normal=normal, ratio=ratio)
        fshape = tuple(int(round((h - l) / self.field_dx)) + 2 for l, h in zip(self.origin, self.hi))
        f = von_karman(fshape, self.field_dx, corr_len, hurst, seed, normal=normal, ratio=ratio)
        idx = np.meshgrid(*[np.arange(n) * self.dx / self.field_dx for n in self.shape], indexing="ij")
        return map_coordinates(f, idx, order=1, mode="nearest").astype(np.float32)

    def axis(self, i: int) -> np.ndarray:
        return self.origin[i] + self.dx * np.arange(self.shape[i])

    def coords(self):
        """Sparse coordinate arrays (S, N, V); V = 0.0 in 2-D."""
        g = np.meshgrid(*[self.axis(i) for i in range(self.dims)], indexing="ij", sparse=True)
        return (g[0], g[1], g[2] if self.dims == 3 else 0.0)

    def to_index(self, pos: np.ndarray, comp: int | None = None):
        """Nearest staggered node: comp None = stress / pressure node, comp d = velocity along dim d."""
        pos = np.atleast_2d(pos)
        off = np.zeros(self.dims)
        if comp is not None:
            off[comp] = 0.5
        idx = np.round((pos - self.origin) / self.dx - off).astype(int)
        idx = np.clip(idx, 0, np.array(self.shape) - 2)
        return idx, self.origin + (idx + off) * self.dx

    def extent(self, i: int):
        return self.origin[i], self.origin[i] + self.dx * (self.shape[i] - 1)


def make_domain(params: dict, meta: dict, dims: int) -> tuple[Domain, TunnelGeom]:
    geom = TunnelGeom(params["tunnel"], dims)
    d, acq = meta["domain"], params["acquisition"]
    s_all = [p["s"] for p in acq["receivers"] + acq["shots"] + acq["virtual"]]
    s_min = min(s_all) - d["behind_first_receiver"]
    s_max = geom.face_s + d["ahead_of_face"]
    ss = np.linspace(s_min, s_max, 400)
    n_wall = geom.n_c(ss) + geom.R
    lo = [s_min, n_wall.min() - d["cavity_margin"]]
    hi = [s_max, n_wall.max() + d["depth_into_rock"]]
    if dims == 3:
        vc = geom.v_c(ss)
        lo.append(vc.min() - d["vertical_half"])
        hi.append(vc.max() + d["vertical_half"])
    dx = float(meta["grid"]["dx_3d" if dims == 3 else "dx_2d"])
    lo, hi = np.array(lo), np.array(hi)
    shape = tuple(int(np.ceil((h - l) / dx)) + 1 for l, h in zip(lo, hi))
    return Domain(origin=lo, dx=dx, shape=shape, dims=dims, hi=hi, field_dx=meta["grid"].get("field_dx")), geom


# --------------------------------------------------------------------------- #
def _anchor_point(anchor: dict, geom3: TunnelGeom) -> np.ndarray:
    """Anchor in full 3-D coordinates. Geology is always defined in 3-D; a 2-D model is its v = 0 slice."""
    return geom3.wall_point(anchor["s"], anchor["offset"], anchor["height"])


def _signed_distance(S, N, V, p0, normal):
    return (S - p0[0]) * normal[0] + (N - p0[1]) * normal[1] + (V - p0[2]) * normal[2]


def paint_rock(params: dict, meta: dict, dom: Domain, geom: TunnelGeom) -> dict:
    """Host rock, structures, texture and density; no tunnel yet."""
    rp = params["rock"]
    geom = TunnelGeom(params["tunnel"], 3)       # 3-D geometry for anchors, also when painting a 2-D slice
    S, N, V = dom.coords()
    shape = dom.shape
    h = rp["host"]
    vp = np.full(shape, h["vp"], np.float32)
    vs = np.full(shape, h["vs"], np.float32)
    dens = np.full(shape, h["density_factor"], np.float32)
    labels = np.zeros(shape, np.int8)
    clay = np.zeros(shape, bool)
    void = np.zeros(shape, bool)

    for st in sorted(rp["structures"], key=lambda x: STRUCT_ORDER.index(x["type"])):
        typ = st["type"]
        if typ in ("interface", "fault_zone", "dyke"):
            d = _signed_distance(S, N, V, _anchor_point(st["anchor"], geom), st["normal"])
        if typ == "interface":
            m = np.broadcast_to(d > 0, shape)
            L = st["lithology"]
            vp[m], vs[m], dens[m] = L["vp"], L["vs"], L["density_factor"]
            labels[m] = LABELS["interface"]
        elif typ == "gradient":
            g = st["direction"]
            proj = np.broadcast_to(S * g[0] + N * g[1] + V * g[2], shape)
            t = (proj - proj.min()) / (proj.max() - proj.min() + 1e-12)
            f = (1.0 - st["contrast"] * t).astype(np.float32)
            vp *= f
            vs *= f
        elif typ == "fault_zone":
            ad = np.broadcast_to(np.abs(d), shape)
            w2 = 0.5 * st["width"]
            halo = w2 * st["halo_factor"]
            c = st["contrast"]
            f = np.ones(shape, np.float32)
            core = ad < w2
            f[core] = 1.0 - c
            if halo > w2:
                hm = (~core) & (ad < halo)
                f[hm] = 1.0 - c * (1.0 - (ad[hm] - w2) / (halo - w2))
                labels[hm & (labels == 0)] = LABELS["fault_halo"]
            labels[core] = LABELS["fault_core"]
            vp *= f
            vs *= f
            dens *= 1.0 - 0.3 * (1.0 - f)
        elif typ == "dyke":
            m = np.broadcast_to(np.abs(d) < 0.5 * st["width"], shape)
            c = st["contrast"]
            vp[m] *= 1.0 - c
            vs[m] *= 1.0 - c
            dens[m] *= 1.0 - 0.5 * c
            labels[m] = LABELS["dyke"]
        elif typ == "karst":
            for vd in st["voids"]:
                p0 = _anchor_point(vd["anchor"], geom)
                a = vd["semi_axes"]
                e = ((S - p0[0]) / a[0]) ** 2 + ((N - p0[1]) / a[1]) ** 2 + ((V - p0[2]) / a[2]) ** 2
                m = np.broadcast_to(e < 1.0, shape)
                if vd["fill"] == "empty":
                    void |= m
                else:
                    clay |= m

    # texture (multiplicative, same field on vp and vs)
    tx = rp["texture"]
    T = np.zeros(shape, np.float32)
    for scale, normal, ratio in (("large", rp["fabric_normal"], h["fabric_ratio"]), ("small", None, 1.0)):
        t = tx[scale]
        if t["std"] > 0:
            T += t["std"] * dom.field(t["corr_length"], t["hurst"], t["field_seed"], normal=normal, ratio=ratio)
    vp *= 1.0 + T
    vs *= 1.0 + T
    rho = (gardner_rho(vp) * dens).astype(np.float32)

    vp[clay], vs[clay], rho[clay] = 1800.0, 900.0, 1900.0
    labels[clay] = LABELS["karst_clay"]
    vs_floor = float(meta["grid"]["vs_floor"])
    np.maximum(vs, vs_floor, out=vs)
    np.maximum(vp, 1.45 * vs, out=vp)
    vp[void] = vs[void] = rho[void] = 0.0
    labels[void] = LABELS["karst_void"]
    return {"vp": vp, "vs": vs, "rho": rho, "labels": labels, "void": void}


def tunnel_geometry_fields(params: dict, dom: Domain, geom: TunnelGeom):
    """Radius r, rock-surface radius r_rock, 'before the face' mask, and the roughness object."""
    S, N, V = dom.coords()
    dn = N - geom.n_c(S)
    dv = (V - geom.v_c(S)) if dom.dims == 3 else np.zeros(1, np.float32)
    r = np.sqrt(dn ** 2 + dv ** 2)
    theta = np.arctan2(dv, dn)
    s_lo, s_hi = dom.extent(0)
    s_lo, s_hi = (dom.origin[0], dom.hi[0]) if dom.hi is not None else (s_lo, s_hi)
    rough = Roughness(params["tunnel"]["roughness"], geom.R, s_lo - 1.0, s_hi + 1.0, dom.field_dx or min(dom.dx, 0.1))
    r_rock = geom.R + rough(np.broadcast_to(S, r.shape), np.broadcast_to(theta, r.shape))
    before_face = np.broadcast_to(S <= geom.face_s, r.shape)
    return np.broadcast_to(r, dom.shape), np.broadcast_to(r_rock, dom.shape), \
        np.broadcast_to(before_face, dom.shape), rough, np.broadcast_to(S, dom.shape)


def build_model(params: dict, meta: dict, dims: int, wall: str, dom: Domain | None = None,
                geom: TunnelGeom | None = None, rock: dict | None = None, tfields=None) -> dict:
    """vp / vs / rho / labels for one wall variant ('concrete' or 'bare').

    `rock` and `tfields` can be passed in to share work between the wall variants.
    """
    if dom is None:
        dom, geom = make_domain(params, meta, dims)
    rock = rock if rock is not None else paint_rock(params, meta, dom, geom)
    r, r_rock, before_face, rough, S = tfields if tfields is not None else tunnel_geometry_fields(params, dom, geom)
    vp, vs, rho, labels = (rock[k].copy() for k in ("vp", "vs", "rho", "labels"))

    excavated = (r < r_rock) & before_face
    # excavation damage zone: distance to the excavated volume (wall and face)
    edz_p = params["rock"]["edz"]
    d = np.sqrt(np.maximum(S - geom.face_s, 0.0) ** 2 + np.maximum(r - r_rock, 0.0) ** 2)
    edz = (~excavated) & (d < edz_p["thickness"]) & (vp > 0)
    f = edz_p["factor"]
    vp[edz] *= f
    vs[edz] *= f
    rho[edz] *= 1.0 - 0.3 * (1.0 - f)
    labels[edz] = LABELS["edz"]

    if wall == "concrete":
        c = params["concrete"]
        # smooth inner surface (circle) clearing the deepest rock protrusion inside the box by `thickness`
        r_in = float(r_rock[before_face].min()) - c["thickness"]
        conc = excavated & (r >= r_in)
        cavity = excavated & (r < r_in)
        noise = 1.0 + c["noise_std"] * dom.field(c["noise_corr_length"], 0.5, c["field_seed"])
        vp[conc] = c["vp"] * noise[conc]
        vs[conc] = c["vp"] / c["vp_vs"] * noise[conc]
        rho[conc] = c["rho"]
        labels[conc] = LABELS["concrete"]
    elif wall == "bare":
        cavity = excavated
    else:
        raise ValueError(f"unknown wall variant {wall!r}")
    vp[cavity] = vs[cavity] = rho[cavity] = 0.0
    labels[cavity] = LABELS["cavity"]
    return {"vp": vp, "vs": vs, "rho": rho, "labels": labels, "cavity": cavity | rock["void"],
            "excavated": excavated, "wall": wall}


def build_variants(params: dict, meta: dict, dims: int, walls) -> tuple[Domain, TunnelGeom, dict]:
    dom, geom = make_domain(params, meta, dims)
    rock = paint_rock(params, meta, dom, geom)
    tf = tunnel_geometry_fields(params, dom, geom)
    models = {w: build_model(params, meta, dims, w, dom, geom, rock, tf) for w in walls}
    return dom, geom, models
