"""
SASA: Self-disciplined Autoregressive Sampling

An implementation of the SASA algorithm from:
"Large Language Models can be Strong Self-Detoxifiers" (arXiv:2410.03818)

This package provides:
- SubspaceLearner: Learn toxic/non-toxic subspaces from embeddings
- MultiLayerSubspaceLearner: Per-layer subspaces with selection/ensembling
- SASASampler: Generate text with toxicity reduction
- BaselineSampler: Standard sampling for comparison
- SASALogitsProcessor: SASA steering inside `model.generate()`
- BackboneAdapter / HFTransformerAdapter / MambaAdapter / JambaAdapter:
  architecture-agnostic model access
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
from .logits_processor import SASALogitsProcessor
from .adapters import (
    AdapterOutput,
    BackboneAdapter,
    HFTransformerAdapter,
    MambaAdapter,
    JambaAdapter
)
from .multilayer import (
    MultiLayerSubspaceLearner,
    extract_embeddings_multilayer
)

__version__ = "0.4.0"

__all__ = [
    "SubspaceLearner",
    "SubspaceParams",
    "extract_embeddings_from_model",
    "SASASampler",
    "BaselineSampler",
    "SASALogitsProcessor",
    "AdapterOutput",
    "BackboneAdapter",
    "HFTransformerAdapter",
    "MambaAdapter",
    "JambaAdapter",
    "MultiLayerSubspaceLearner",
    "extract_embeddings_multilayer",
]
