"""
Backbone adapter layer for SASA.

SASA's core algorithm needs only two things from a model: per-step logits and a
readable hidden state at the last position. This module defines a narrow
`BackboneAdapter` protocol exposing exactly that, so `SASASampler` can run on
any architecture that implements it — decoder-only Transformers, SSMs (Mamba),
hybrids (Jamba), encoder-decoder decoders, etc.

Architecture applicability (see issue #34):
- AR Transformers / SSMs / hybrids / MoE: full support (this module).
- Encoder-only masked LMs: no autoregressive decoding; SubspaceLearner only.
- Diffusion LMs: require a denoising-step analogue (issue #42), which will
  implement `supports_incremental() == False` with a step-based interface.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Protocol, Any, runtime_checkable


@dataclass
class AdapterOutput:
    """Result of one adapter forward step.

    Attributes:
        logits: Logits for the last position, shape (vocab_size,).
        hidden_state: Final-layer hidden state at the last position,
            shape (embedding_dim,).
        past_key_values: Cache to pass to the next step, or None if the
            backbone does not support caching.
    """
    logits: torch.Tensor
    hidden_state: torch.Tensor
    past_key_values: Any = None


@runtime_checkable
class BackboneAdapter(Protocol):
    """Minimal interface SASA requires from a language-model backbone."""

    def forward_step(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> AdapterOutput:
        """Run one forward step and return last-position logits + hidden state.

        Args:
            input_ids: Token ids. When past_key_values is provided, only the
                newest token(s) need to be passed.
            past_key_values: Cache returned by the previous step, or None.
            use_cache: Request cache output from the backbone.
        """
        ...

    def token_embeddings(self) -> torch.Tensor:
        """Input embedding matrix, shape (vocab_size, embedding_dim)."""
        ...

    def supports_kv_cache(self) -> bool:
        """Whether the backbone returns reusable per-step caches."""
        ...

    def supports_incremental(self) -> bool:
        """Whether left-to-right incremental decoding is defined.

        False for diffusion/masked LMs, which need a denoising-step interface
        (issue #42) rather than this protocol.
        """
        ...


class HFTransformerAdapter:
    """BackboneAdapter for HuggingFace causal LMs (GPT-2, Llama, Mamba,
    Jamba, and any AutoModelForCausalLM exposing hidden states).

    For Mamba/Jamba, HF returns a recurrent-state cache rather than a KV
    cache; this adapter passes it through transparently and reports
    `supports_kv_cache()` based on whether the model actually returns one.
    """

    def __init__(self, model):
        self._model = model
        self._cache_seen: Optional[bool] = None

    def forward_step(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> AdapterOutput:
        outputs = self._model(
            input_ids,
            past_key_values=past_key_values if use_cache else None,
            output_hidden_states=True,
            use_cache=use_cache,
        )
        cache = getattr(outputs, "past_key_values", None) if use_cache else None
        if use_cache and self._cache_seen is None:
            self._cache_seen = cache is not None
        return AdapterOutput(
            logits=outputs.logits[0, -1, :],
            hidden_state=outputs.hidden_states[-1][0, -1, :],
            past_key_values=cache,
        )

    def token_embeddings(self) -> torch.Tensor:
        return self._model.get_input_embeddings().weight

    def supports_kv_cache(self) -> bool:
        # Unknown until the first step; True is the safe optimistic default
        # because forward_step falls back to cache=None transparently.
        return self._cache_seen if self._cache_seen is not None else True

    def supports_incremental(self) -> bool:
        return True
