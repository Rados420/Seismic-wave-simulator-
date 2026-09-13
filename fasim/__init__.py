"""fasim: prototype 2-D (plan-view) elastic tunnel-seismic simulator.

Coordinate convention (everywhere in this package):
    dim 0 = z_t  along-tunnel coordinate [m], face direction positive
    dim 1 = x_t  across-tunnel coordinate [m], 0 = tunnel centreline,
                 right wall at +half_width, left wall at -half_width
Deepwave's "y" axis is our dim 0 (z_t) and its "x" axis is our dim 1 (x_t).
"""
from .config import ExperimentCfg, load_config, save_config  # noqa: F401
