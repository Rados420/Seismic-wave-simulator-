"""V1 dataset generator: sampler (meta -> random draw), model builder (2-D / 3-D), simulation driver."""
from .meta import load_meta  # noqa: F401
from .sampler import sample_draw, summary  # noqa: F401
