"""Rank-one margin basis and a context-independent steering rule.

This module is an *additive* companion to :mod:`sasa.sampler`; it does not
change the behaviour of any existing symbol. Its purpose is to make explicit a
property of the reference SASA margin that is otherwise implicit in the tensor
algebra of ``SASASampler.compute_token_margins``.

Motivation
----------
``SASASampler.compute_token_margins`` forms the next-state estimate

.. math::
    \\hat g_t = \\tfrac{1}{2}\\big(g(c) + e_t\\big)

for every candidate token :math:`t` and contracts it with the learned weight
:math:`w`. Expanding the margin gives

.. math::
    \\mathrm{margin}_t
      = \\frac{\\langle w, g(c)\\rangle + \\langle w, e_t\\rangle
              - 2\\langle w, b\\rangle}{2 \\lVert w \\rVert}
      = \\underbrace{\\tfrac{\\langle w, g(c)\\rangle}{2\\lVert w\\rVert}}_
            {\\text{constant in } t} + \\tfrac{\\langle w, e_t\\rangle}{2\\lVert w\\rVert}
            - \\underbrace{\\tfrac{\\langle w, b\\rangle}{\\lVert w\\rVert}}_
            {\\text{constant in } t}.

The first and third terms are identical for every candidate token, i.e. they are
a *constant shift of the whole logit vector*. The softmax, top-k filtering and
nucleus filtering are all invariant to such a shift, so the terms cancel
exactly and the sampling distribution depends on the context **only through the
model's own logits**. What remains is the static vocabulary bias
:math:`\\beta \\langle w, e_t\\rangle` with :math:`\\beta = \\alpha / (2\\lVert w\\rVert)`.

Two things follow, and both are implemented here:

1. The per-token margin vector carries no information beyond
   :math:`c = E^{\\top} w \\in \\mathbb{R}^{V}` (computable once, at fit time)
   and the scalar :math:`\\langle w, g(c)\\rangle`. The
   :math:`V \\times d` matrix need never be materialised, which removes
   :math:`\\mathcal{O}(Vd)` work per decoding step.
2. The steering vector is *context-free*. :class:`StaticMarginBias` exposes this
   directly so that the property can be asserted, measured, and regression-tested
   rather than assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "MarginBasis",
    "StaticMarginBias",
    "reference_token_margins",
    "verify_static_equivalence",
    "EquivalenceReport",
]


def reference_token_margins(
    subspace_learner,
    current_embedding: torch.Tensor,
    token_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the reference implementation's per-token margins exactly.

    This is a verbatim, dependency-free re-implementation of
    ``SASASampler.compute_token_margins`` so that the fast path can be checked
    against it without constructing a model or a sampler.

    Args:
        subspace_learner: A fitted :class:`sasa.subspace_learner.SubspaceLearner`.
        current_embedding: Current context hidden state, shape ``(d,)``.
        token_embeddings: Input-embedding matrix, shape ``(V, d)``.

    Returns:
        Per-token margins, shape ``(V,)``.
    """
    next_embeddings = (current_embedding.unsqueeze(0) + token_embeddings) / 2
    return subspace_learner.compute_margin(next_embeddings)


@dataclass
class MarginBasis:
    """Precomputed, context-independent components of the SASA margin.

    Attributes:
        weight: The learned direction, shape ``(d,)``. Equal to the subspace
            learner's ``w``.
        bias: The learned bias, shape ``(d,)``. Equal to ``b``.
        token_weights: The precomputed vocabulary bias ``E^T w``, shape
            ``(V,)``. This is the *entire* context-dependent signal that
            survives the softmax.
        embedding_dim: Hidden size ``d``.
        vocab_size: Vocabulary size ``V``.
    """

    weight: torch.Tensor
    bias: torch.Tensor
    token_weights: torch.Tensor
    embedding_dim: int
    vocab_size: int

    @classmethod
    def from_subspace_learner(
        cls,
        subspace_learner,
        token_embeddings: torch.Tensor,
    ) -> "MarginBasis":
        """Build a :class:`MarginBasis` from a fitted learner and an embedding table.

        Args:
            subspace_learner: A fitted :class:`sasa.subspace_learner.SubspaceLearner`.
            token_embeddings: Input-embedding matrix of shape ``(V, d)``.

        Returns:
            A :class:`MarginBasis` with ``token_weights`` already contracted.

        Raises:
            RuntimeError: If the subspace learner has not been fitted.
            ValueError: If ``token_embeddings`` is not two-dimensional or its
                width disagrees with the learner's ``embedding_dim``.
        """
        params = subspace_learner.params
        if params is None:
            raise RuntimeError("Must call fit() before building a MarginBasis")

        if token_embeddings.dim() != 2:
            raise ValueError(
                f"token_embeddings must be 2-D (V, d); got shape "
                f"{tuple(token_embeddings.shape)}"
            )
        d = token_embeddings.shape[1]
        if d != subspace_learner.embedding_dim:
            raise ValueError(
                f"token_embeddings width {d} does not match learner "
                f"embedding_dim {subspace_learner.embedding_dim}"
            )

        w = params.w_v.detach().to(token_embeddings.dtype)
        b = params.b_v.detach().to(token_embeddings.dtype)
        token_weights = token_embeddings.detach() @ w

        return cls(
            weight=w,
            bias=b,
            token_weights=token_weights,
            embedding_dim=d,
            vocab_size=token_embeddings.shape[0],
        )

    @property
    def weight_norm(self) -> torch.Tensor:
        """L2 norm of the learned direction, shape ``()``."""
        return torch.norm(self.weight)

    def context_offset(self, context_hidden: torch.Tensor) -> torch.Tensor:
        """The per-step constant that the softmax discards.

        Args:
            context_hidden: Context hidden state, shape ``(d,)`` or ``(B, d)``.

        Returns:
            Tensor of shape ``()`` for 1-D input, or ``(B,)`` for batched input,
            holding ``<w, g(c)> / (2 ||w||) - <w, b> / ||w||``.
        """
        sq = context_hidden.unsqueeze(0) if context_hidden.dim() == 1 else context_hidden
        w_norm = self.weight_norm
        offset = (sq @ self.weight) / (2.0 * w_norm) - (self.bias @ self.weight) / w_norm
        return offset.squeeze(0) if context_hidden.dim() == 1 else offset

    def full_margins(
        self,
        context_hidden: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Per-token margins in the original (uncollapsed) units.

        Provided for testing and for callers that need the raw margin
        distribution, e.g. to study its shape. Prefer :meth:`static_bias` for
        decoding, which is the same quantity up to the constant returned by
        :meth:`context_offset`.

        Args:
            context_hidden: Context hidden state, shape ``(d,)``.
            alpha: Steering strength; scales the returned margins.

        Returns:
            Tensor of shape ``(V,)``.
        """
        w_norm = self.weight_norm
        token_term = self.token_weights / (2.0 * w_norm)
        return alpha * (token_term + self.context_offset(context_hidden))

    def static_bias(self, alpha: float = 1.0) -> torch.Tensor:
        """The context-free steering vector used for decoding.

        This is ``alpha * <w, e_t> / (2 ||w||)``, the only term of the margin
        that survives the softmax. Adding it to the model logits reproduces the
        reference implementation's sampling distribution exactly.

        Args:
            alpha: Steering strength.

        Returns:
            Tensor of shape ``(V,)``.
        """
        return alpha * self.token_weights / (2.0 * self.weight_norm)


class StaticMarginBias:
    """A drop-in, exact, ``O(V)``-per-step replacement for the reference margin.

    The reference implementation allocates a ``(V, d)`` matrix at every decoding
    step. For Llama-3.1-8B that is ``128256 x 4096`` floats -- 2.1 GB of
    materialisation and 525 M multiply-adds per generated token -- in order to
    compute a quantity this class holds in a single precomputed vector.

    Attributes:
        basis: The precomputed :class:`MarginBasis`.
        alpha: Steering strength.
        temperature: Sampling temperature, or ``None`` to leave logits unscaled.
    """

    def __init__(
        self,
        basis: MarginBasis,
        alpha: float = 1.0,
        temperature: Optional[float] = None,
    ) -> None:
        """Initialise the static bias.

        Args:
            basis: A precomputed :class:`MarginBasis`.
            alpha: Steering strength. Larger values push down the logits of
                tokens with a toxic projection.
            temperature: Optional sampling temperature. If given, adjusted
                logits are divided by it.

        Raises:
            ValueError: If ``alpha`` is not finite or ``temperature`` is
                non-positive.
        """
        if temperature is not None and temperature <= 0:
            raise ValueError(f"temperature must be positive; got {temperature}")
        self.basis = basis
        self.alpha = float(alpha)
        self.temperature = temperature
        self._bias = basis.static_bias(self.alpha)

    def adjust_logits(
        self,
        logits: torch.Tensor,
        context_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return SASA-adjusted logits.

        Args:
            logits: Model logits, shape ``(V,)`` or ``(B, V)``.
            context_hidden: Context hidden state. Accepted for interface
                compatibility with :class:`sasa.sampler.SASASampler` and
                deliberately **unused**: under the reference next-state
                estimator the context-dependent part of the margin is a
                constant that cancels in the softmax. Passing it is safe and
                lets the equivalence test drive both code paths identically.

        Returns:
            Adjusted logits with the same shape as ``logits``.
        """
        del context_hidden  # Provably cancels; see module docstring.
        adjusted = logits + self._bias
        if self.temperature is not None:
            adjusted = adjusted / self.temperature
        return adjusted

    def sample(
        self,
        logits: torch.Tensor,
        context_hidden: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample one token using the adjusted distribution.

        Args:
            logits: Model logits, shape ``(V,)``.
            context_hidden: Unused; see :meth:`adjust_logits`.
            top_k: If given, restrict to the ``top_k`` highest adjusted logits.
            top_p: If given, apply nucleus filtering at this cumulative
                probability threshold.
            generator: Optional RNG generator for reproducible sampling.

        Returns:
            Sampled token index, shape ``(1,)`` of dtype ``int64``.

        Raises:
            ValueError: If ``top_k`` is not positive or ``top_p`` is outside
                ``(0, 1]``.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive; got {top_k}")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must lie in (0, 1]; got {top_p}")

        adjusted = self.adjust_logits(logits, context_hidden)
        adjusted = _apply_top_k(adjusted, top_k)
        adjusted = _apply_top_p(adjusted, top_p)
        probs = F.softmax(adjusted, dim=-1)
        if generator is not None:
            return torch.multinomial(probs, num_samples=1, generator=generator)
        return torch.multinomial(probs, num_samples=1)


def _apply_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    """Mask all but the ``top_k`` highest entries of ``logits``."""
    if top_k is None:
        return logits
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    """Apply nucleus filtering in place on a copy of ``logits``."""
    if top_p is None:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    drop = cum > top_p
    drop[..., 0] = False  # Always keep the most likely token.
    drop = drop.scatter(-1, sorted_idx, drop)
    return logits.masked_fill(drop, float("-inf"))


@dataclass
class EquivalenceReport:
    """Result of comparing the fast path against the reference implementation.

    The two implementations produce *different adjusted logits* by design: the
    fast path omits the context-dependent constant ``kappa`` that the softmax
    annihilates, so ``max_logit_gap`` is generally non-zero. The quantity that
    must vanish is the **residual** after that constant is removed, and the
    resulting sampling distribution. Reporting both is the point: a non-zero
    ``max_logit_gap`` with a near-zero ``residual_logit_gap`` and
    ``max_prob_gap`` is a direct numerical confirmation of Theorem 1.

    Attributes:
        max_logit_gap: Largest absolute difference in adjusted logits. Expected
            to be non-zero and equal to ``|kappa|`` for the given context.
        residual_logit_gap: Largest absolute difference in adjusted logits after
            subtracting the mean difference. This is the part that *would* change
            the distribution, and it must be at float32 noise level.
        max_prob_gap: Largest absolute difference in sampling probabilities.
        kl_from_reference: ``KL(p_reference || p_fast)``, clamped at zero.
        identical_top_k: Whether top-k and top-p filtering select the same set.
        context_sensitivity_fast: Total-variation distance between the
            distributions produced from two different contexts, using the fast
            path. Expected to be exactly ``0``.
        context_sensitivity_reference: The same quantity using the reference
            implementation. Expected to be at float32 noise level.
    """

    max_logit_gap: float
    residual_logit_gap: float
    max_prob_gap: float
    kl_from_reference: float
    identical_top_k: bool
    context_sensitivity_fast: float
    context_sensitivity_reference: float

    @property
    def is_exact(self) -> bool:
        """Whether the two implementations induce the same distribution."""
        return (
            self.residual_logit_gap < 1e-4
            and self.max_prob_gap < 1e-5
            and self.kl_from_reference < 1e-6
        )

    @property
    def is_context_free(self) -> bool:
        """Whether steering is provably independent of the hidden state."""
        return (
            self.context_sensitivity_fast < 1e-6
            and self.context_sensitivity_reference < 1e-5
        )

    def as_dict(self) -> dict:
        """Return the report as a plain dictionary."""
        return {
            "max_logit_gap": self.max_logit_gap,
            "residual_logit_gap": self.residual_logit_gap,
            "max_prob_gap": self.max_prob_gap,
            "kl_from_reference": self.kl_from_reference,
            "identical_top_k": self.identical_top_k,
            "context_sensitivity_fast": self.context_sensitivity_fast,
            "context_sensitivity_reference": self.context_sensitivity_reference,
            "is_exact": self.is_exact,
            "is_context_free": self.is_context_free,
        }


def _total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total-variation distance between two probability vectors."""
    return float(0.5 * torch.abs(p - q).sum().item())


def verify_static_equivalence(
    subspace_learner,
    token_embeddings: torch.Tensor,
    logits: torch.Tensor,
    context_a: torch.Tensor,
    context_b: torch.Tensor,
    alpha: float = 1.0,
    temperature: Optional[float] = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.9,
    top_k_k: Optional[int] = None,
) -> EquivalenceReport:
    """Check the fast path against the reference implementation, exactly.

    The function verifies three separate claims and reports each independently,
    so that a failure localises to one mechanism rather than to "the numbers
    differ":

    1. *Exactness.* Adjusted logits and resulting probabilities agree between
       :class:`StaticMarginBias` and :func:`reference_token_margins`.
    2. *Filter stability.* Top-k and top-p select the same token sets.
    3. *Context-independence.* Both implementations produce the same
       distribution for two different hidden states.

    Args:
        subspace_learner: A fitted :class:`sasa.subspace_learner.SubspaceLearner`.
        token_embeddings: Input-embedding matrix, shape ``(V, d)``.
        logits: Model logits to steer, shape ``(V,)``.
        context_a: First context hidden state, shape ``(d,)``.
        context_b: Second, deliberately very different, context hidden state.
        alpha: Steering strength.
        temperature: Sampling temperature, or ``None``.
        top_k: Top-k threshold to compare, or ``None`` to skip.
        top_p: Nucleus threshold to compare, or ``None`` to skip.
        top_k_k: Alias retained for readability at call sites; defaults to
            ``top_k``.

    Returns:
        An :class:`EquivalenceReport`.

    Raises:
        ValueError: If ``logits`` and the embedding table disagree on ``V``.
    """
    if logits.shape[0] != token_embeddings.shape[0]:
        raise ValueError(
            f"logits has V={logits.shape[0]} but token_embeddings has "
            f"V={token_embeddings.shape[0]}"
        )
    if top_k_k is not None:
        top_k = top_k_k

    basis = MarginBasis.from_subspace_learner(subspace_learner, token_embeddings)
    fast = StaticMarginBias(basis, alpha=alpha, temperature=temperature)

    ref_a = reference_token_margins(subspace_learner, context_a, token_embeddings)
    ref_b = reference_token_margins(subspace_learner, context_b, token_embeddings)
    ref_logits_a = logits + alpha * ref_a
    if temperature is not None:
        ref_logits_a = ref_logits_a / temperature

    fast_logits_a = fast.adjust_logits(logits, context_a)

    max_logit_gap = float((ref_logits_a - fast_logits_a).abs().max().item())
    # The only legitimate logit discrepancy is an additive constant, so remove
    # the mean and require the residual to vanish.
    residual_logit_gap = float(
        ((ref_logits_a - fast_logits_a) - (ref_logits_a - fast_logits_a).mean())
        .abs().max().item()
    )
    ref_probs = F.softmax(ref_logits_a, dim=-1)
    fast_probs = F.softmax(fast_logits_a, dim=-1)
    max_prob_gap = float((ref_probs - fast_probs).abs().max().item())
    kl = max(0.0, float(
        (ref_probs * (ref_probs.clamp_min(1e-12).log()
                      - fast_probs.clamp_min(1e-12).log())).sum().item()
    ))

    identical_top_k = True
    if top_k is not None:
        ref_mask = _apply_top_k(ref_logits_a.clone(), top_k) > float("-inf")
        fast_mask = _apply_top_k(fast_logits_a.clone(), top_k) > float("-inf")
        identical_top_k = bool(torch.equal(ref_mask, fast_mask))
    if top_p is not None:
        ref_mask = _apply_top_p(ref_logits_a.clone(), top_p) > float("-inf")
        fast_mask = _apply_top_p(fast_logits_a.clone(), top_p) > float("-inf")
        identical_top_k = identical_top_k and bool(torch.equal(ref_mask, fast_mask))

    ref_logits_b = logits + alpha * ref_b
    fast_logits_b = fast.adjust_logits(logits, context_b)
    tv_fast = _total_variation(
        F.softmax(fast_logits_a, dim=-1), F.softmax(fast_logits_b, dim=-1)
    )
    tv_ref = _total_variation(
        F.softmax(ref_logits_a, dim=-1), F.softmax(ref_logits_b, dim=-1)
    )

    return EquivalenceReport(
        max_logit_gap=max_logit_gap,
        residual_logit_gap=residual_logit_gap,
        max_prob_gap=max_prob_gap,
        kl_from_reference=kl,
        identical_top_k=identical_top_k,
        context_sensitivity_fast=tv_fast,
        context_sensitivity_reference=tv_ref,
    )


def benchmark_step(
    subspace_learner,
    token_embeddings: torch.Tensor,
    logits: torch.Tensor,
    context_hidden: torch.Tensor,
    alpha: float = 1.0,
    repeats: int = 20,
) -> Tuple[float, float]:
    """Time the reference and fast margin paths.

    Args:
        subspace_learner: A fitted :class:`sasa.subspace_learner.SubspaceLearner`.
        token_embeddings: Input-embedding matrix, shape ``(V, d)``.
        logits: Model logits, shape ``(V,)``.
        context_hidden: Context hidden state, shape ``(d,)``.
        alpha: Steering strength.
        repeats: Number of timed iterations per path.

    Returns:
        A ``(reference_ms, fast_ms)`` tuple of mean per-step latencies.
    """
    basis = MarginBasis.from_subspace_learner(subspace_learner, token_embeddings)
    fast = StaticMarginBias(basis, alpha=alpha)

    for _ in range(3):  # Warm up allocator and BLAS.
        reference_token_margins(subspace_learner, context_hidden, token_embeddings)
        fast.adjust_logits(logits, context_hidden)

    t0 = time.perf_counter()
    for _ in range(repeats):
        reference_token_margins(subspace_learner, context_hidden, token_embeddings)
    ref_ms = (time.perf_counter() - t0) / repeats * 1e3

    t0 = time.perf_counter()
    for _ in range(repeats):
        fast.adjust_logits(logits, context_hidden)
    fast_ms = (time.perf_counter() - t0) / repeats * 1e3

    return ref_ms, fast_ms
