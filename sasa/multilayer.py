"""
Multi-Layer Subspace Learning for SASA (issue #39)

The base `SubspaceLearner` fits the toxic/non-toxic subspace on a single hidden
layer (the last one). Intermediate layers often show cleaner linear separability
for toxicity. This module fits one subspace per layer, ranks layers by held-out
separability, and supports two decode-time modes:

- ``best``: use only the selected layer's subspace (cheap; pair with
  ``HFTransformerAdapter(hidden_layer=k)``).
- ``ensemble``: average the per-layer margins (robust; pair with
  ``HFTransformerAdapter(include_all_layers=True)``). Ensemble margins across
  layers are also harder for jailbreaks to fool simultaneously (see #45).
"""

import torch
from typing import Dict, List, Optional
from .subspace_learner import SubspaceLearner


class MultiLayerSubspaceLearner:
    """
    Fits and selects among per-layer toxic/non-toxic subspaces.

    Attributes:
        embedding_dim: Hidden dimension of the model.
        layers: Layer indices (into HF ``hidden_states``) to fit.
        selection: 'best' (single selected layer) or 'ensemble' (mean margin).
        learners: Mapping layer index -> fitted SubspaceLearner.
        selected_layer: Layer chosen by separability validation (best mode).
    """

    def __init__(
        self,
        embedding_dim: int,
        layers: List[int],
        selection: str = "best",
    ):
        if selection not in ("best", "ensemble"):
            raise ValueError("selection must be 'best' or 'ensemble'")
        if not layers:
            raise ValueError("layers must be a non-empty list of layer indices")
        self.embedding_dim = embedding_dim
        self.layers = list(layers)
        self.selection = selection
        self.learners: Dict[int, SubspaceLearner] = {
            l: SubspaceLearner(embedding_dim) for l in self.layers
        }
        self.selected_layer: Optional[int] = None

    def fit(
        self,
        embeddings_non_toxic: Dict[int, torch.Tensor],
        embeddings_toxic: Dict[int, torch.Tensor],
    ) -> "MultiLayerSubspaceLearner":
        """
        Fit one subspace per layer.

        Args:
            embeddings_non_toxic: layer index -> (N1, embedding_dim) tensor.
            embeddings_toxic: layer index -> (N2, embedding_dim) tensor.
        """
        for l in self.layers:
            self.learners[l].fit(embeddings_non_toxic[l], embeddings_toxic[l])
        return self

    def separability_scores(
        self,
        val_non_toxic: Dict[int, torch.Tensor],
        val_toxic: Dict[int, torch.Tensor],
    ) -> Dict[int, float]:
        """
        Held-out linear separability (classification accuracy) per layer.

        Positive margins are expected for non-toxic embeddings and negative
        margins for toxic ones; accuracy is the fraction correct.
        """
        scores: Dict[int, float] = {}
        for l in self.layers:
            learner = self.learners[l]
            m_nt = learner.compute_margin(val_non_toxic[l])
            m_t = learner.compute_margin(val_toxic[l])
            correct = (m_nt > 0).float().mean() * len(m_nt) + (m_t < 0).float().mean() * len(m_t)
            scores[l] = (correct / (len(m_nt) + len(m_t))).item()
        return scores

    def select_layer(
        self,
        val_non_toxic: Dict[int, torch.Tensor],
        val_toxic: Dict[int, torch.Tensor],
    ) -> int:
        """Pick the layer with the highest held-out separability."""
        scores = self.separability_scores(val_non_toxic, val_toxic)
        self.selected_layer = max(scores, key=scores.get)
        return self.selected_layer

    def compute_margin(self, embeddings) -> torch.Tensor:
        """
        Margin(s) for embeddings.

        Args:
            embeddings: In 'best' mode, a single embedding (..., embedding_dim)
                from the selected layer. In 'ensemble' mode, a dict mapping
                layer index -> embedding, or a stacked tensor whose first
                dimension indexes ``self.layers``.

        Returns:
            Margin values (mean across layers in ensemble mode).
        """
        if self.selection == "ensemble":
            if isinstance(embeddings, dict):
                per_layer = [self.learners[l].compute_margin(embeddings[l]) for l in self.layers]
            else:
                per_layer = [
                    self.learners[l].compute_margin(embeddings[i])
                    for i, l in enumerate(self.layers)
                ]
            return torch.stack(per_layer, dim=0).mean(dim=0)

        layer = self.selected_layer if self.selected_layer is not None else self.layers[-1]
        if isinstance(embeddings, dict):
            return self.learners[layer].compute_margin(embeddings[layer])
        return self.learners[layer].compute_margin(embeddings)

    def classify(self, embedding: torch.Tensor) -> torch.Tensor:
        """Classification score (positive = non-toxic) in best mode."""
        layer = self.selected_layer if self.selected_layer is not None else self.layers[-1]
        return self.learners[layer].classify(embedding)

    def save(self, path: str) -> None:
        payload = {
            "embedding_dim": self.embedding_dim,
            "layers": self.layers,
            "selection": self.selection,
            "selected_layer": self.selected_layer,
            "learners": {},
        }
        for l, learner in self.learners.items():
            if learner.params is None:
                raise RuntimeError(f"Layer {l} learner is not fitted")
            payload["learners"][l] = {
                "w_v": learner.params.w_v,
                "b_v": learner.params.b_v,
                "mu_1": learner.params.mu_1,
                "mu_2": learner.params.mu_2,
                "sigma": learner.params.sigma,
            }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path)
        self.embedding_dim = payload["embedding_dim"]
        self.layers = payload["layers"]
        self.selection = payload["selection"]
        self.selected_layer = payload["selected_layer"]
        self.learners = {}
        for l, params in payload["learners"].items():
            learner = SubspaceLearner(self.embedding_dim)
            # Reuse fit math via a tiny synthetic re-fit is wasteful; set params directly.
            from .subspace_learner import SubspaceParams
            learner.params = SubspaceParams(
                w_v=params["w_v"],
                b_v=params["b_v"],
                mu_1=params["mu_1"],
                mu_2=params["mu_2"],
                sigma=params["sigma"],
                embedding_dim=self.embedding_dim,
            )
            self.learners[l] = learner


def extract_embeddings_multilayer(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    layers: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Extract last-token hidden states at multiple layers for a list of texts.

    Returns:
        Dict mapping layer index -> (len(texts), embedding_dim) tensor.
    """
    model.eval()
    collected: Dict[int, List[torch.Tensor]] = {}

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple: (n_layers + 1) tensors
            use_layers = layers if layers is not None else list(range(len(hidden_states)))
            for l in use_layers:
                idx = l if l >= 0 else len(hidden_states) + l
                collected.setdefault(l, []).append(hidden_states[idx][0, -1, :].cpu())

    return {l: torch.stack(v) for l, v in collected.items()}
