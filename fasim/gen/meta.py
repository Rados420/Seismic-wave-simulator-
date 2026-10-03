"""Dataset metaparameters: YAML loading, command-line overrides, distribution sampling."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")


def load_meta(path: str | Path | None = None, overrides: list[str] | tuple = ()) -> dict:
    """Defaults, then the user file deep-merged on top, then `key.path=value` overrides."""
    with open(DEFAULTS_PATH) as fh:
        meta = yaml.safe_load(fh)
    if path is not None:
        with open(path) as fh:
            deep_merge(meta, yaml.safe_load(fh) or {})
    for ov in overrides:
        key, _, val = ov.partition("=")
        if not _:
            raise ValueError(f"override must look like key.path=value, got {ov!r}")
        set_path(meta, key.strip(), yaml.safe_load(val))
    return meta


def is_dist(spec) -> bool:
    return isinstance(spec, dict) and "dist" in spec


def deep_merge(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and not is_dist(v) and isinstance(base.get(k), dict) and not is_dist(base[k]):
            deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def set_path(d: dict, path: str, value) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def draw(spec, rng: np.random.Generator):
    """Sample a value from a distribution spec; scalars and lists are returned unchanged."""
    if not is_dist(spec):
        return spec
    d = spec["dist"]
    if d == "uniform":
        x = rng.uniform(spec["low"], spec["high"])
    elif d == "loguniform":
        x = np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"])))
    elif d == "normal":
        x = rng.normal(spec["mean"], spec["std"])
    elif d == "lognormal":
        x = spec["median"] * np.exp(spec["sigma"] * rng.standard_normal())
    elif d == "randint":
        x = int(rng.integers(spec["low"], spec["high"] + 1))
    elif d == "poisson":
        x = int(rng.poisson(spec["lam"]))
    elif d == "beta":
        lo, hi = spec.get("low", 0.0), spec.get("high", 1.0)
        x = lo + (hi - lo) * rng.beta(spec["a"], spec["b"])
    elif d == "choice":
        vals = spec["values"]
        w = np.asarray(spec.get("weights", [1.0] * len(vals)), float)
        return vals[int(rng.choice(len(vals), p=w / w.sum()))]
    else:
        raise ValueError(f"unknown distribution {d!r}")
    if "clip" in spec:
        x = np.clip(x, *spec["clip"])
    return int(x) if isinstance(x, (int, np.integer)) else float(x)


def uniform_range(r, rng: np.random.Generator) -> float:
    """[lo, hi] -> uniform draw; a scalar is returned as is."""
    if isinstance(r, (list, tuple)):
        return float(rng.uniform(r[0], r[1]))
    return float(r)


def lerp(pair, t: float) -> float:
    """pair = [value at t = 1, value at t = 0]."""
    return float(pair[1] + (pair[0] - pair[1]) * t)
