"""Draw one dataset sample: every random parameter of one draw, as a JSON-serialisable dict.

All parameters are drawn independently. Each component (tunnel, concrete, acquisition,
sources, rock) has its own random stream derived from (seed_base, seed, component), so
changing how one component is sampled does not change the others. Random fields are stored
as integer seeds, so the dict alone reproduces the models exactly.

The enumerated factors (wall variant, source type) are not drawn: the simulator runs all of
their combinations on top of the same draw.
"""
from __future__ import annotations

import numpy as np

from .meta import draw, lerp, uniform_range

STREAMS = ("tunnel", "concrete", "acquisition", "sources", "rock")


def _rngs(meta: dict, seed: int) -> dict[str, np.random.Generator]:
    base = int(meta.get("seed_base", 0))
    return {name: np.random.default_rng([base, int(seed), i]) for i, name in enumerate(STREAMS)}


def _field_seed(rng) -> int:
    return int(rng.integers(0, 2 ** 31 - 1))


def _unit_vector(rng) -> list[float]:
    v = rng.standard_normal(3)
    return (v / np.linalg.norm(v)).tolist()


def sample_draw(meta: dict, seed: int) -> dict:
    r = _rngs(meta, seed)
    acq = sample_acquisition(meta, r["acquisition"])
    tunnel = sample_tunnel(meta, r["tunnel"], acq)
    params = {
        "seed": int(seed),
        "seed_base": int(meta.get("seed_base", 0)),
        "tunnel": tunnel,
        "concrete": sample_concrete(meta, r["concrete"]),
        "acquisition": acq,
        "sources": sample_sources(meta, r["sources"], len(acq["shots"])),
    }
    params["rock"] = sample_rock(meta, r["rock"], acq, tunnel)
    return params


# --------------------------------------------------------------------------- #
def sample_acquisition(meta: dict, rng) -> dict:
    a = meta["acquisition"]
    n_r = int(draw(a["n_receivers"], rng))
    sp_r = draw(a["receiver_spacing"], rng)
    gap = draw(a["gap"], rng)
    n_s = int(draw(a["n_shots"], rng))
    sp_s = draw(a["shot_spacing"], rng)
    j_al, j_dp, j_ht = (draw(a[k], rng) for k in ("jitter_along", "jitter_depth", "jitter_height"))
    h0 = draw(a["height"], rng)

    # per-position deviations in their own streams: changing a count does not shift other draws
    rng_r, rng_s = (np.random.default_rng(_field_seed(rng)) for _ in range(2))

    def point(g, s_nom, embed_nom):
        return {"s": float(s_nom + j_al * g.standard_normal()),
                "embed": float(max(0.3, embed_nom + j_dp * g.standard_normal())),
                "height": float(h0 + j_ht * g.standard_normal())}

    rcv = [point(rng_r, i * sp_r, a["receiver_embed"]) for i in range(n_r)]
    s_first_shot = (n_r - 1) * sp_r + gap
    shots = [point(rng_s, s_first_shot + k * sp_s, a["shot_embed"]) for k in range(n_s)]
    virtual = []
    vr = a.get("virtual_receivers", {})
    if vr.get("enabled"):
        s_end = s_first_shot + (n_s - 1) * sp_s
        virtual = [{"s": float(s), "embed": float(a["receiver_embed"]), "height": float(h0)}
                   for s in np.arange(0.0, s_end + 1e-9, vr["spacing"])]
    return {"n_receivers": n_r, "receiver_spacing": sp_r, "gap": gap, "n_shots": n_s, "shot_spacing": sp_s,
            "jitter_along": j_al, "jitter_depth": j_dp, "jitter_height": j_ht, "height": h0,
            "receivers": rcv, "shots": shots, "virtual": virtual}


def sample_tunnel(meta: dict, rng, acq: dict) -> dict:
    t = meta["tunnel"]
    c = t["curvature"]
    sv, sh, rho = float(c["std_vertical"]), float(c["std_horizontal"]), float(c["correlation"])
    z = rng.standard_normal(2)
    kv = sv * z[0]
    kh = sh * (rho * z[0] + np.sqrt(max(1.0 - rho ** 2, 0.0)) * z[1])
    s_last_shot = (acq["n_receivers"] - 1) * acq["receiver_spacing"] + acq["gap"] + \
        (acq["n_shots"] - 1) * acq["shot_spacing"]
    ro = t["roughness"]
    return {
        "radius": float(draw(t["radius"], rng)),
        "curvature_vertical": float(kv),
        "curvature_horizontal": float(kh),       # > 0: the tunnel bends towards the receiver wall
        "s_ref": float(s_last_shot / 2.0),       # the wall passes through n = 0 here
        "face_s": float(s_last_shot + draw(t["face_distance"], rng)),
        "roughness": {"amplitude": draw(ro["amplitude"], rng), "corr_length": draw(ro["corr_length"], rng),
                      "overbreak_bias": draw(ro["overbreak_bias"], rng), "field_seed": _field_seed(rng)},
    }


def sample_concrete(meta: dict, rng) -> dict:
    c = meta["concrete"]
    out = {k: draw(c[k], rng) for k in ("thickness", "vp", "vp_vs", "rho", "noise_std", "noise_corr_length")}
    out["field_seed"] = _field_seed(rng)
    return out


def sample_sources(meta: dict, rng, n_shots: int) -> dict:
    s = meta["sources"]
    ex = {"peak_freq": draw(s["explosive"]["peak_freq"], rng), "amplitude": draw(s["explosive"]["amplitude"], rng)}
    im = s["impact"]
    f_im, a_im, cone = (draw(im[k], rng) for k in ("peak_freq", "amplitude", "direction_jitter_deg"))
    shot_rng = np.random.default_rng(_field_seed(rng))     # per-shot draws last, in their own stream
    ca = np.cos(np.deg2rad(cone))
    polar = np.rad2deg(np.arccos(1.0 - shot_rng.uniform(size=n_shots) * (1.0 - ca)))   # uniform on the cone cap
    azim = shot_rng.uniform(0.0, 360.0, size=n_shots)
    imp = {"peak_freq": f_im, "amplitude": a_im,
           "direction_jitter_deg": cone, "polar_deg": polar.tolist(), "azimuth_deg": azim.tolist()}
    return {"explosive": ex, "impact": imp}


# --------------------------------------------------------------------------- #
def _lithology(meta: dict, rng, quality: float, exclude: str | None = None) -> dict:
    rk = meta["rock"]
    names = [n for n in rk["lithologies"] if n != exclude] or list(rk["lithologies"])
    w = np.array([rk["lithologies"][n]["weight"] for n in names], float)
    name = names[int(rng.choice(len(names), p=w / w.sum()))]
    L = rk["lithologies"][name]
    qe = rk["quality_effects"]
    vp_i = uniform_range(L["vp"], rng)
    vp_vs = uniform_range(L["vp_vs"], rng)
    fabric = L.get("fabric")
    return {
        "name": name,
        "vp_intact": vp_i,
        "vp_vs_intact": vp_vs,
        "vp": vp_i * (1.0 - qe["vp_reduction_max"] * (1.0 - quality)),
        "vs": vp_i / vp_vs * (1.0 - qe["vs_reduction_max"] * (1.0 - quality)),
        "density_factor": float(1.0 + rk["density_scatter"] * rng.standard_normal()),
        "q_p": uniform_range(L["q"], rng) * (0.5 + 0.5 * quality),
        "fabric": fabric,
        "fabric_ratio": float(draw(rk["fabric_anisotropy"][fabric], rng)) if fabric else 1.0,
        "karst": bool(L.get("karst", False)),
    }


def _anchor(rng, rk: dict, acq: dict, tunnel: dict, meta: dict) -> dict:
    """Point a structure passes through, in wall-relative coordinates (s, offset from wall, height)."""
    s_lo = min(p["s"] for p in acq["receivers"])
    s_hi = max(p["s"] for p in acq["shots"])
    d = meta["domain"]
    if rng.uniform() < rk["structures"]["in_aperture_prob"]:
        return {"s": float(rng.uniform(s_lo, s_hi)), "offset": float(rng.uniform(0.0, 6.0)),
                "height": float(rng.uniform(-2.0, 2.0)), "in_aperture": True}
    return {"s": float(rng.uniform(s_lo - d["behind_first_receiver"], tunnel["face_s"] + d["ahead_of_face"])),
            "offset": float(rng.uniform(0.0, d["depth_into_rock"])),
            "height": float(rng.uniform(-d["vertical_half"], d["vertical_half"])), "in_aperture": False}


def sample_rock(meta: dict, rng, acq: dict, tunnel: dict) -> dict:
    rk = meta["rock"]
    quality = float(draw(rk["quality"], rng))
    host = _lithology(meta, rng, quality)
    qe = rk["quality_effects"]
    edz = {"thickness": lerp(qe["edz_thickness"], quality), "factor": lerp(qe["edz_factor"], quality)}

    st_cfg = rk["structures"]
    n_struct = min(int(draw(st_cfg["count"], rng)), int(st_cfg["max_count"]))
    types = {k: v for k, v in st_cfg["types"].items() if k != "karst" or host["karst"]}
    tnames = list(types)
    tw = np.array([types[k]["weight"] for k in tnames], float)
    structures = []
    for _ in range(n_struct):
        typ = tnames[int(rng.choice(len(tnames), p=tw / tw.sum()))]
        c = types[typ]
        st = {"type": typ}
        if typ in ("fault_zone", "dyke", "interface"):
            st["anchor"] = _anchor(rng, rk, acq, tunnel, meta)
            st["normal"] = _unit_vector(rng)
        if typ == "fault_zone":
            st.update(width=draw(c["width"], rng), contrast=draw(c["contrast"], rng),
                      halo_factor=draw(c["halo_factor"], rng))
        elif typ == "dyke":
            st.update(width=draw(c["width"], rng), contrast=draw(c["contrast"], rng))
        elif typ == "interface":
            st["lithology"] = _lithology(meta, rng, quality)
        elif typ == "karst":
            n = int(draw(c["count"], rng))
            st["voids"] = []
            for _ in range(n):
                rad = draw(c["radius"], rng)
                st["voids"].append({"anchor": _anchor(rng, rk, acq, tunnel, meta),
                                    "semi_axes": (rad * rng.uniform(0.5, 1.5, size=3)).tolist(),
                                    "fill": draw(c["fill"], rng)})
        elif typ == "gradient":
            st.update(direction=_unit_vector(rng), contrast=draw(c["contrast"], rng))
        structures.append(st)

    tex = {}
    for scale in ("large", "small"):
        t = rk["texture"][scale]
        tex[scale] = {k: draw(t[k], rng) for k in ("std", "corr_length", "hurst")}
        tex[scale]["field_seed"] = _field_seed(rng)
    return {"quality": quality, "host": host, "edz": edz, "fabric_normal": _unit_vector(rng),
            "structures": structures, "texture": tex}


def summary(params: dict) -> str:
    """One-screen human-readable summary of a draw."""
    t, a, r, s, c = params["tunnel"], params["acquisition"], params["rock"], params["sources"], params["concrete"]
    def radius(k):
        return "straight" if abs(k) < 1e-6 else f"R {1 / abs(k):.0f} m ({'+' if k > 0 else '-'})"
    h = r["host"]
    lines = [
        f"seed {params['seed']}",
        f"tunnel  : radius {t['radius']} m, horizontal {radius(t['curvature_horizontal'])}, "
        f"vertical {radius(t['curvature_vertical'])}, face at s = {t['face_s']:.1f} m",
        f"wall    : roughness {t['roughness']['amplitude']:.2f} m rms / {t['roughness']['corr_length']:.1f} m, "
        f"overbreak bias {t['roughness']['overbreak_bias']:.2f}; concrete {c['thickness']:.2f} m, vp {c['vp']:.0f}",
        f"acq     : {a['n_receivers']} receivers @ {a['receiver_spacing']:.2f} m, gap {a['gap']:.1f} m, "
        f"{a['n_shots']} shots @ {a['shot_spacing']:.2f} m, jitter {a['jitter_along']:.2f}/{a['jitter_depth']:.2f}/"
        f"{a['jitter_height']:.2f} m, {len(a['virtual'])} virtual",
        f"sources : explosive {s['explosive']['peak_freq']:.0f} Hz, impact {s['impact']['peak_freq']:.0f} Hz "
        f"(cone {s['impact']['direction_jitter_deg']:.1f} deg)",
        f"rock    : {h['name']} q={r['quality']:.2f} vp {h['vp']:.0f} vs {h['vs']:.0f} Q_P {h['q_p']:.0f} "
        f"fabric {h['fabric'] or '-'} x{h['fabric_ratio']:.1f}; EDZ {r['edz']['thickness']:.2f} m x{r['edz']['factor']:.2f}",
        f"texture : large {r['texture']['large']['std']:.3f}/{r['texture']['large']['corr_length']:.1f} m, "
        f"small {r['texture']['small']['std']:.3f}/{r['texture']['small']['corr_length']:.2f} m",
    ]
    for st in r["structures"]:
        extra = ""
        if st["type"] in ("fault_zone", "dyke"):
            extra = f"width {st['width']:.2f} m, contrast {st['contrast']:+.2f}"
        elif st["type"] == "interface":
            extra = f"-> {st['lithology']['name']} vp {st['lithology']['vp']:.0f}"
        elif st["type"] == "karst":
            extra = f"{len(st['voids'])} voids, fills {[v['fill'] for v in st['voids']]}"
        elif st["type"] == "gradient":
            extra = f"contrast {st['contrast']:.2f}"
        where = ""
        if "anchor" in st:
            where = f" at s={st['anchor']['s']:.1f} ({'aperture' if st['anchor']['in_aperture'] else 'elsewhere'})"
        lines.append(f"  struct: {st['type']}{where} {extra}")
    return "\n".join(lines)
