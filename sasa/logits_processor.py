"""
SASA HuggingFace LogitsProcessor

Wraps SASA margin-based steering as a `transformers.LogitsProcessor`, so it
composes with `model.generate()` (beam search, top-k/top-p warpers, standard
decoding) instead of requiring the custom sampling loop in `SASASampler`.

The processor captures the last-position hidden state at every generation step
via a forward pre-hook on the model's output embedding (lm_head), then adds
`alpha * margin` to the logits — exactly the SASA adjustment.

Example:
    processor = SASALogitsProcessor(learner, model, alpha=1.0, margin_top_k=50)
    output = model.generate(**inputs, max_new_tokens=50,
                            logits_processor=LogitsProcessorList([processor]))
    processor.close()  # removes the hook
"""

import torch
from typing import Optional

from transformers import LogitsProcessor

from .subspace_learner import SubspaceLearner


class SASALogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that applies SASA margin steering inside `model.generate()`.

    Args:
        subspace_learner: Trained SubspaceLearner instance.
        model: A HuggingFace causal LM (used to capture hidden states and to
            access token embeddings).
        alpha: Weight for margin-based steering (higher = stronger
            detoxification).
        margin_top_k: If set, only the top-k tokens by logit receive a margin
            adjustment; all other tokens get zero margin.
    """

    def __init__(
        self,
        subspace_learner: SubspaceLearner,
        model,
        alpha: float = 1.0,
        margin_top_k: Optional[int] = None,
    ):
        self.subspace_learner = subspace_learner
        self.alpha = alpha
        self.margin_top_k = margin_top_k
        self._token_embeddings = model.get_input_embeddings().weight
        self._hidden: Optional[torch.Tensor] = None

        # Capture the input to lm_head — i.e. the final hidden states — on
        # every forward pass (prompt and each cached decode step).
        self._hook = model.get_output_embeddings().register_forward_pre_hook(
            self._capture_hidden
        )

    def _capture_hidden(self, module, args):
        hidden = args[0]  # (batch, seq_len, hidden_dim)
        self._hidden = hidden[:, -1, :].detach()

    def _margins_for(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Margins for one sequence: hidden (hidden_dim,), logits (vocab_size,)."""
        vocab_size = logits.shape[-1]
        if self.margin_top_k is not None and self.margin_top_k < vocab_size:
            top_indices = torch.topk(logits, self.margin_top_k).indices
            margins = torch.zeros_like(logits)
            next_embeddings = (hidden.unsqueeze(0) + self._token_embeddings[top_indices]) / 2
            margins[top_indices] = self.subspace_learner.compute_margin(next_embeddings)
            return margins
        next_embeddings = (hidden.unsqueeze(0) + self._token_embeddings) / 2
        return self.subspace_learner.compute_margin(next_embeddings)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self._hidden is None:
            # Hook has not fired (unexpected); return scores unchanged.
            return scores

        adjusted = scores.clone()
        for batch_idx in range(scores.shape[0]):
            margins = self._margins_for(self._hidden[batch_idx], scores[batch_idx])
            adjusted[batch_idx] = scores[batch_idx] + self.alpha * margins
        return adjusted

    def close(self):
        """Remove the hidden-state capture hook. Call when generation is done."""
        self._hook.remove()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
