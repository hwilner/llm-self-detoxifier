"""
SASA: Self-disciplined Autoregressive Sampling

An implementation of the SASA algorithm from:
"Large Language Models can be Strong Self-Detoxifiers" (arXiv:2410.03818)

This package provides:
- SubspaceLearner: Learn toxic/non-toxic subspaces from embeddings
- SASASampler: Generate text with toxicity reduction
- BaselineSampler: Standard sampling for comparison
"""

from .subspace_learner import (
    SubspaceLearner,
    SubspaceParams,
    extract_embeddings_from_model
)
from .sampler import (
    SASASampler,
    BaselineSampler
)

__version__ = "0.1.0"

__all__ = [
    "SubspaceLearner",
    "SubspaceParams",
    "extract_embeddings_from_model",
    "SASASampler",
    "BaselineSampler",
]
