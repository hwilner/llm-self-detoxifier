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
from dataclasses import dataclass, field
from typing import Optional, Protocol, Any, Tuple, runtime_checkable


@dataclass
class AdapterOutput:
    """Result of one adapter forward step.

    Attributes:
        logits: Logits for the last position, shape (vocab_size,).
        hidden_state: Hidden state at the last position from the configured
            layer, shape (embedding_dim,).
        past_key_values: Cache to pass to the next step, or None if the
            backbone does not support caching.
        hidden_states: All-layer hidden states at the last position (tuple of
            tensors, each shape (embedding_dim,)), populated only when the
            adapter was created with include_all_layers=True. Used by
            MultiLayerSubspaceLearner in ensemble mode (issue #39).
    """
    logits: torch.Tensor
    hidden_state: torch.Tensor
    past_key_values: Any = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


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

    Args:
        model: A HuggingFace causal LM.
        hidden_layer: Which entry of ``outputs.hidden_states`` to expose as
            ``AdapterOutput.hidden_state`` (default -1, the final layer).
            Pair with ``MultiLayerSubspaceLearner(selection='best')`` to steer
            from the most separable layer (issue #39).
        include_all_layers: If True, also populate ``AdapterOutput.hidden_states``
            with every layer's last-position hidden state. Required for
            ``MultiLayerSubspaceLearner(selection='ensemble')``.
    """

    def __init__(self, model, hidden_layer: int = -1, include_all_layers: bool = False):
        self._model = model
        self.hidden_layer = hidden_layer
        self.include_all_layers = include_all_layers
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
        # HF Transformers return `past_key_values`; Mamba/SSM implementations
        # return a recurrent-state cache under `cache_params`. Accept both so
        # incremental decoding works across architectures transparently.
        cache = None
        if use_cache:
            cache = getattr(outputs, "past_key_values", None)
            if cache is None:
                cache = getattr(outputs, "cache_params", None)
            if self._cache_seen is None:
                self._cache_seen = cache is not None

        hs = outputs.hidden_states
        all_layers = None
        if self.include_all_layers:
            all_layers = tuple(h[0, -1, :] for h in hs)

        idx = self.hidden_layer if self.hidden_layer >= 0 else len(hs) + self.hidden_layer
        return AdapterOutput(
            logits=outputs.logits[0, -1, :],
            hidden_state=hs[idx][0, -1, :],
            past_key_values=cache,
            hidden_states=all_layers,
        )

    def token_embeddings(self) -> torch.Tensor:
        return self._model.get_input_embeddings().weight

    def supports_kv_cache(self) -> bool:
        # Unknown until the first step; True is the safe optimistic default
        # because forward_step falls back to cache=None transparently.
        return self._cache_seen if self._cache_seen is not None else True

    def supports_incremental(self) -> bool:
        return True


class MambaAdapter(HFTransformerAdapter):
    """BackboneAdapter for Mamba / Mamba-2 selective state-space models
    (e.g. ``state-spaces/mamba-130m-hf``).

    Everything is inherited from HFTransformerAdapter; the HF Mamba
    implementation exposes ``cache_params`` (conv + SSM recurrent state),
    which the base class picks up automatically. Toxicity may be less
    linearly separable in SSM states than in Transformer activations — run
    ``experiments/separability_probe.py`` before choosing ``hidden_layer``
    (issue #40).
    """

    @classmethod
    def from_pretrained(cls, model_name: str = "state-spaces/mamba-130m-hf", **kwargs):
        from transformers import AutoModelForCausalLM
        return cls(AutoModelForCausalLM.from_pretrained(model_name, **kwargs))


class JambaAdapter(HFTransformerAdapter):
    """BackboneAdapter for AI21 Jamba (Transformer/Mamba/MoE hybrid).

    Recommended default: steer from the final hidden state (hidden_layer=-1),
    which is architecture-agnostic and directly comparable to Llama results.
    Steering from a Transformer block's output rather than an SSM block's is
    the alternative worth probing with ``experiments/separability_probe.py``.
    Note: Jamba MoE routing does not change the hidden-state interface.
    """

    @classmethod
    def from_pretrained(cls, model_name: str = "ai21labs/Jamba-v0.1", **kwargs):
        from transformers import AutoModelForCausalLM
        return cls(AutoModelForCausalLM.from_pretrained(model_name, **kwargs))
