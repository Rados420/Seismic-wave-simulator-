"""Configuration dataclasses + YAML round trip.

Every knob of the prototype lives here so a run is fully described by one
YAML file (stored verbatim in the output HDF5 for reproducibility).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Lithology lookup: rough ranges, used as defaults / documentation.
# Rock "composition" is only vp, vp/vs and rho (Gardner if not given).
# --------------------------------------------------------------------------- #
LITHOLOGY = {
    "granite":    dict(vp=(5500.0, 6000.0), vp_vs=(1.65, 1.85)),
    "gneiss":     dict(vp=(5000.0, 6000.0), vp_vs=(1.65, 1.90)),
    "limestone":  dict(vp=(4500.0, 6000.0), vp_vs=(1.75, 1.95)),
    "fault_zone": dict(vp=(2500.0, 4000.0), vp_vs=(1.80, 2.10)),
}


def gardner_rho(vp: Any) -> Any:
    """Gardner's relation, vp in m/s -> rho in kg/m^3 (0.31 * vp^0.25 g/cc)."""
    return 310.0 * vp ** 0.25


@dataclass
class DomainCfg:
    z_min: float = -10.0      # along-tunnel extent [m]
    z_max: float = 70.0
    x_min: float = -25.0      # across-tunnel extent [m]
    x_max: float = 25.0
    dx: float = 0.2           # grid spacing [m], same in both dims
    pml_width: int = 20       # absorbing cells on every domain edge


@dataclass
class RockCfg:
    """Intact host rock."""
    vp: float = 5800.0
    vp_vs: float = 1.75
    rho: float | None = None          # None -> Gardner
    texture_std: float = 0.0          # relative std of multiplicative texture on vp, vs (0 = off)
    texture_corr_len: float = 2.0     # correlation length of the texture [m]


@dataclass
class TunnelCfg:
    enabled: bool = True
    half_width: float = 5.25          # tunnel half width in plan view [m]
    z_start: float | None = None      # None -> cavity runs out through the back boundary
    z_face: float = 60.0              # along-tunnel position of the face [m]
    edz_thickness: float = 1.0        # excavation damage zone thickness [m]; 0 disables
    edz_factor: float = 0.7           # multiplies vp and vs inside the EDZ
    edz_rho_factor: float = 0.9       # multiplies rho inside the EDZ


@dataclass
class StructureCfg:
    """Large-scale geology ahead of / around the tunnel."""
    family: str = "homogeneous"       # homogeneous | fault_zone | layered_dip
    # fault_zone: band of low velocity crossing the tunnel axis
    fault_z: float = 75.0             # where the band centre crosses x_t = 0 [m]
    fault_width: float = 5.0          # band thickness measured normal to the band [m]
    fault_strike_deg: float = 90.0    # angle between band and tunnel axis in plan view (90 = perpendicular)
    fault_vp: float = 3500.0
    fault_vp_vs: float = 1.9
    # layered_dip: everything beyond a straight interface gets new properties
    layer_z: float = 40.0             # where the interface crosses x_t = 0 [m]
    layer_strike_deg: float = 60.0    # angle between interface and tunnel axis
    layer_vp: float = 4500.0
    layer_vp_vs: float = 1.8


@dataclass
class ReceiverCfg:
    z: float                          # along-tunnel station [m]
    side: str = "left"                # left | right wall
    embed: float = 1.5                # borehole depth into the wall [m]
    name: str = ""


@dataclass
class AcquisitionCfg:
    receivers: list[ReceiverCfg] = field(default_factory=lambda: [
        ReceiverCfg(z=7.0, side="right", embed=1.5, name="RCV1"),
        ReceiverCfg(z=2.0, side="right", embed=1.5, name="RCV2"),
        ReceiverCfg(z=7.0, side="left", embed=1.5, name="RCV3"),
        ReceiverCfg(z=2.0, side="left", embed=1.5, name="RCV4"),
    ])
    shot_side: str = "right"
    shot_z0: float = 22.0             # first shot along-tunnel position [m]
    shot_dz: float = 1.5              # shot spacing [m]
    n_shots: int = 24
    shot_embed: float = 1.5           # shot-hole depth into the wall [m]
    shot_z: list[float] | None = None  # explicit list overrides z0/dz/n


@dataclass
class SourceCfg:
    kind: str = "force"               # force (hammer) | pressure (explosive)
    direction_deg: float = 0.0        # force direction: 0 = wall normal into the rock, 90 = along tunnel (+z_t)
    peak_freq: float = 600.0          # Ricker peak frequency [Hz]
    amplitude: float = 1.0


@dataclass
class SimCfg:
    dt: float = 20.833e-6             # output sample interval [s] (48 kHz, like the field data)
    record_ms: float = 30.0           # record length [ms]
    accuracy: int = 4                 # Deepwave FD order (2 or 4)
    device: str = "cpu"
    shot_chunk: int = 8               # shots per Deepwave call (memory vs speed)
    snapshots_ms: list[float] = field(default_factory=list)  # wavefield snapshot times for shot 0
    snapshot_shot: int = 0


@dataclass
class LabelCfg:
    picker: str = "amplitude_threshold"  # see fasim.onset.PICKERS; envelope_threshold is non-causal (Hilbert tails of later phases)
    rel_threshold: float = 0.02          # threshold as a fraction of the reference amplitude
    threshold_ref: str = "window"        # reference amplitude: trace | gather | window
                                         #   trace  = max of the whole trace (S / guided waves dominate)
                                         #   gather = max over the whole receiver gather (mimics a fixed noise floor)
                                         #   window = max within ref_window_ms after the search start (P-wave shape emergence)
    ref_window_ms: float = 2.0
    component: str = "x"                 # component defining the primary onset: y | x | p | vec
    search_before_ms: float = 1.0        # search starts this long before eikonal + wavelet onset (None = whole trace)
    sta_ms: float = 0.1
    lta_ms: float = 1.0
    trigger: float = 5.0


@dataclass
class PostCfg:
    """Amplitude post-processing (see fasim.postprocess). null / 0 = off."""
    q: float | None = None            # constant-Q intrinsic attenuation (hard rock ~50-300)
    spreading_exponent: float = 0.0   # extra (t/1ms)^-p decay; 0.5 mimics 3-D spreading


@dataclass
class ExperimentCfg:
    name: str = "homogeneous"
    seed: int = 0
    domain: DomainCfg = field(default_factory=DomainCfg)
    rock: RockCfg = field(default_factory=RockCfg)
    tunnel: TunnelCfg = field(default_factory=TunnelCfg)
    structure: StructureCfg = field(default_factory=StructureCfg)
    acquisition: AcquisitionCfg = field(default_factory=AcquisitionCfg)
    source: SourceCfg = field(default_factory=SourceCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    labels: LabelCfg = field(default_factory=LabelCfg)
    post: PostCfg = field(default_factory=PostCfg)


# --------------------------------------------------------------------------- #
# dict <-> dataclass
# --------------------------------------------------------------------------- #
def _from_dict(cls, d: dict):
    if d is None:
        return cls()
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if f.name == "receivers":
            v = [ReceiverCfg(**r) if isinstance(r, dict) else r for r in v]
        elif f.name in _NESTED and isinstance(v, dict):
            v = _from_dict(_NESTED[f.name], v)
        kwargs[f.name] = v
    unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)


_NESTED = dict(domain=DomainCfg, rock=RockCfg, tunnel=TunnelCfg, structure=StructureCfg,
               acquisition=AcquisitionCfg, source=SourceCfg, sim=SimCfg, labels=LabelCfg, post=PostCfg)


def to_dict(cfg: ExperimentCfg) -> dict:
    return dataclasses.asdict(cfg)


def from_dict(d: dict) -> ExperimentCfg:
    return _from_dict(ExperimentCfg, d)


def load_config(path: str | Path) -> ExperimentCfg:
    with open(path) as fh:
        return from_dict(yaml.safe_load(fh) or {})


def save_config(cfg: ExperimentCfg, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)
