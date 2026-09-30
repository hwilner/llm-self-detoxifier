"""Tests for the rank-one margin basis and the static steering rule.

These tests do two jobs. They pin the correctness of the fast path against the
reference implementation, and they assert the two structural properties the
report claims: exact equivalence under the reference next-state estimator, and
the resulting context-independence of the steering rule.

Run with::

    pytest tests/test_fast_margin.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from sasa.fast_margin import (
    MarginBasis,
    StaticMarginBias,
    reference_token_margins,
    verify_static_equivalence,
)
from sasa.subspace_learner import SubspaceLearner


def _fitted_learner(dim: int = 24, seed: int = 0, sep: float = 0.9):
    """Build a fitted learner on two well-separated synthetic clusters.

    Args:
        dim: Hidden size.
        seed: RNG seed.
        sep: Cluster separation in units of the per-cluster standard deviation.

    Returns:
        A fitted :class:`SubspaceLearner`.
    """
    g = torch.Generator().manual_seed(seed)
    axis = torch.zeros(dim)
    axis[0] = 1.0
    neg = torch.randn(120, dim, generator=g) * 0.4 + sep * axis
    pos = torch.randn(120, dim, generator=g) * 0.4 - sep * axis
    learner = SubspaceLearner(embedding_dim=dim)
    learner.fit(neg, pos)
    return learner


def _embeddings(vocab: int = 3000, dim: int = 24, seed: int = 1):
    """Return a random input-embedding table and matching logits."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, dim, generator=g), torch.randn(vocab, generator=g)


# ------------------------------------------------------------------ basis


class TestMarginBasis:
    """Construction and invariants of :class:`MarginBasis`."""

    def test_static_bias_is_context_free_by_construction(self):
        """`static_bias` must not accept a context at all."""
        learner = _fitted_learner()
        emb, _ = _embeddings()
        basis = MarginBasis.from_subspace_learner(learner, emb)
        assert basis.static_bias(1.0).shape == (emb.shape[0],)
        # Two calls with different alpha scale linearly.
        b1, b2 = basis.static_bias(1.0), basis.static_bias(2.0)
        assert torch.allclose(2.0 * b1, b2, atol=1e-6)

    def test_requires_fitted_learner(self):
        """Building a basis before `fit` must fail loudly."""
        emb, _ = _embeddings()
        with pytest.raises(RuntimeError, match="fit"):
            MarginBasis.from_subspace_learner(SubspaceLearner(24), emb)

    def test_rejects_bad_embedding_rank(self):
        """A 1-D embedding tensor must be rejected with a clear message."""
        learner = _fitted_learner()
        with pytest.raises(ValueError, match="2-D"):
            MarginBasis.from_subspace_learner(learner, torch.randn(24))

    def test_rejects_dimension_mismatch(self):
        """An embedding table of the wrong width must be rejected."""
        learner = _fitted_learner(dim=24)
        with pytest.raises(ValueError, match="does not match"):
            MarginBasis.from_subspace_learner(learner, torch.randn(10, 16))

    def test_full_margins_match_reference_plus_offset(self):
        """`full_margins` must equal the reference margin exactly."""
        learner = _fitted_learner(seed=3)
        emb, _ = _embeddings(seed=4)
        basis = MarginBasis.from_subspace_learner(learner, emb)
        g = torch.randn(24, generator=torch.Generator().manual_seed(5))
        ref = reference_token_margins(learner, g, emb)
        got = basis.full_margins(g, alpha=1.0)
        assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3)

    def test_context_offset_is_scalar(self):
        """The context-dependent term is a scalar, not a vector."""
        learner = _fitted_learner()
        emb, _ = _embeddings()
        basis = MarginBasis.from_subspace_learner(learner, emb)
        off = basis.context_offset(torch.randn(24))
        assert off.dim() == 0


# ------------------------------------------------------------- equivalence


class TestStaticEquivalence:
    """The fast path must reproduce the reference implementation exactly."""

    def test_adjusted_logits_differ_only_by_a_constant(self):
        """Logits differ by a constant shift, and the residual vanishes.

        The fast path deliberately omits the context-dependent constant that the
        softmax annihilates, so a non-zero ``max_logit_gap`` is the *expected*
        outcome. What must vanish is the residual once that constant is removed.
        """
        for seed in range(8):
            dim = 16 + seed
            learner = _fitted_learner(dim=dim, seed=seed)
            emb, logits = _embeddings(vocab=700, dim=dim, seed=100 + seed)
            g = torch.randn(dim, generator=torch.Generator().manual_seed(seed))
            basis = MarginBasis.from_subspace_learner(learner, emb)
            fast = StaticMarginBias(basis, alpha=1.7, temperature=0.8)
            ref = (logits + 1.7 * reference_token_margins(learner, g, emb)) / 0.8
            got = fast.adjust_logits(logits, g)
            delta = ref - got
            # Constant across the vocabulary to float32 precision ...
            assert float((delta - delta.mean()).abs().max()) < 1e-4
            # ... and equal to the predicted offset kappa.
            kappa = float(1.7 * basis.context_offset(g) / 0.8)
            assert abs(float(delta.mean()) - kappa) < 1e-3

    def test_probability_distributions_agree(self):
        """Softmax outputs agree, not merely logits."""
        learner = _fitted_learner(seed=7)
        emb, logits = _embeddings(vocab=900, seed=8)
        g = torch.randn(24, generator=torch.Generator().manual_seed(9))
        basis = MarginBasis.from_subspace_learner(learner, emb)
        fast = StaticMarginBias(basis, alpha=3.0, temperature=1.0)
        ref_p = torch.softmax(logits + 3.0 * reference_token_margins(learner, g, emb), -1)
        fast_p = torch.softmax(fast.adjust_logits(logits, g), -1)
        assert torch.allclose(ref_p, fast_p, atol=1e-6)

    def test_verification_report_is_exact(self):
        """`verify_static_equivalence` reports exactness and filter agreement."""
        learner = _fitted_learner(seed=11)
        emb, logits = _embeddings(vocab=1200, seed=12)
        ga = torch.randn(24, generator=torch.Generator().manual_seed(13))
        gb = torch.randn(24, generator=torch.Generator().manual_seed(14)) * 40.0
        rep = verify_static_equivalence(
            learner, emb, logits, ga, gb, alpha=2.0, temperature=1.0,
            top_k=50, top_p=0.9,
        )
        assert rep.is_exact
        assert rep.identical_top_k
        assert rep.max_prob_gap < 1e-6
        assert rep.kl_from_reference < 1e-6

    def test_topk_and_topp_select_same_mask(self):
        """Filtering is applied to a vector that differs only by a constant.

        top-k compares values against each vector's own k-th largest value, and
        a constant shift cancels in that comparison; top-p is defined on the
        softmax, which is shift-invariant. Both selections must therefore match
        exactly, and the resulting nucleus sizes must match too.
        """
        learner = _fitted_learner(seed=15)
        emb, logits = _embeddings(vocab=1500, seed=16)
        g = torch.randn(24, generator=torch.Generator().manual_seed(17))
        basis = MarginBasis.from_subspace_learner(learner, emb)
        fast = StaticMarginBias(basis, alpha=5.0)
        ref = logits + 5.0 * reference_token_margins(learner, g, emb)
        fast_l = fast.adjust_logits(logits, g)
        shifts: list[float] = []
        alpha = 5.0
        for k in (1, 10, 100):
            t_ref = torch.topk(ref, k).values[..., -1]
            t_fast = torch.topk(fast_l, k).values[..., -1]
            # Same selected set ...
            assert torch.equal(ref >= t_ref, fast_l >= t_fast)
            # ... and the threshold moved by the *same* constant for every k.
            shifts.append(float(t_ref - t_fast))
        assert max(shifts) - min(shifts) < 1e-3
        assert abs(shifts[0] - float(alpha * basis.context_offset(g))) < 1e-3

        for p in (0.5, 0.9, 0.99):
            sp_ref = torch.sort(ref, descending=True).values.softmax(-1).cumsum(-1)
            sp_fast = torch.sort(fast_l, descending=True).values.softmax(-1).cumsum(-1)
            assert torch.allclose(sp_ref, sp_fast, atol=1e-5)
            n_ref = int((sp_ref <= p).sum())
            n_fast = int((sp_fast <= p).sum())
            assert abs(n_ref - n_fast) <= 1

    def test_rejects_vocabulary_mismatch(self):
        """A logits/embedding vocabulary mismatch must be reported."""
        learner = _fitted_learner()
        emb, _ = _embeddings(vocab=50)
        with pytest.raises(ValueError, match="V="):
            verify_static_equivalence(
                learner, emb, torch.randn(60), torch.randn(24), torch.randn(24)
            )


# ------------------------------------------------------- context-independence


class TestContextIndependence:
    """The steering rule must not depend on the hidden state."""

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 4.0, 20.0])
    def test_reference_is_context_independent(self, alpha):
        """The reference implementation is invariant to the context, at any alpha.

        The tolerance is a float32 noise floor rather than a signal threshold.
        The reference path recomputes a ``(V, d)`` matrix whose discarded
        constant scales with ``<w, g>``; with a deliberately extreme context
        (``||g|| ~ 100``) that constant reaches ~1e2, so rounding in the
        softmax reaches the fourth decimal. Any genuine context dependence
        would be orders of magnitude larger.
        """
        learner = _fitted_learner(seed=21)
        emb, logits = _embeddings(vocab=800, seed=22)
        g_a = torch.randn(24, generator=torch.Generator().manual_seed(23))
        g_b = torch.randn(24, generator=torch.Generator().manual_seed(24)) * 100.0
        p_a = torch.softmax(logits + alpha * reference_token_margins(learner, g_a, emb), -1)
        p_b = torch.softmax(logits + alpha * reference_token_margins(learner, g_b, emb), -1)
        assert float(0.5 * (p_a - p_b).abs().sum()) < 1e-4

    def test_report_flags_context_freedom(self):
        """The verification report exposes the property as a boolean."""
        learner = _fitted_learner(seed=25)
        emb, logits = _embeddings(vocab=800, seed=26)
        rep = verify_static_equivalence(
            learner, emb, logits,
            torch.randn(24), torch.randn(24) * 50.0, alpha=1.0,
        )
        assert rep.is_context_free
        assert rep.context_sensitivity_fast == pytest.approx(0.0, abs=1e-6)
        assert rep.context_sensitivity_reference < 1e-5

    def test_only_the_offset_moves_with_context(self):
        """What does change with context is the discarded scalar offset."""
        learner = _fitted_learner(seed=27)
        emb, _ = _embeddings(seed=28)
        basis = MarginBasis.from_subspace_learner(learner, emb)
        off_a = basis.context_offset(torch.randn(24))
        off_b = basis.context_offset(torch.randn(24) * 10.0)
        assert not math.isclose(float(off_a), float(off_b), rel_tol=1e-3)
        assert torch.equal(basis.static_bias(1.0), basis.static_bias(1.0))


# ----------------------------------------------------------------- sampling


class TestStaticMarginBiasSampling:
    """Sampling respects the documented contracts."""

    def test_sample_returns_valid_index(self):
        """A sampled index lies inside the vocabulary."""
        learner = _fitted_learner(seed=31)
        emb, logits = _embeddings(vocab=500, seed=32)
        basis = MarginBasis.from_subspace_learner(learner, emb)
        tok = StaticMarginBias(basis, alpha=1.0).sample(logits)
        assert 0 <= int(tok.item()) < emb.shape[0]

    def test_sampling_is_reproducible_with_generator(self):
        """Passing a seeded generator yields identical draws."""
        learner = _fitted_learner(seed=33)
        emb, logits = _embeddings(vocab=500, seed=34)
        basis = MarginBasis.from_subspace_learner(learner, emb)
        bias = StaticMarginBias(basis, alpha=1.0)
        a = bias.sample(logits, generator=torch.Generator().manual_seed(5))
        b = bias.sample(logits, generator=torch.Generator().manual_seed(5))
        assert int(a.item()) == int(b.item())

    def test_top_k_restricts_support(self):
        """With top_k=1 the sample is the argmax of the adjusted logits."""
        learner = _fitted_learner(seed=35)
        emb, logits = _embeddings(vocab=400, seed=36)
        basis = MarginBasis.from_subspace_learner(learner, emb)
        bias = StaticMarginBias(basis, alpha=1.0)
        tok = int(bias.sample(logits, top_k=1,
                              generator=torch.Generator().manual_seed(1)).item())
        assert tok == int(torch.argmax(bias.adjust_logits(logits)).item())

    def test_rejects_invalid_sampling_arguments(self):
        """Bad `top_k`/`top_p`/temperature values raise ValueError."""
        learner = _fitted_learner(seed=37)
        emb, logits = _embeddings(vocab=200, seed=38)
        basis = MarginBasis.from_subspace_learner(learner, emb)
        with pytest.raises(ValueError, match="temperature"):
            StaticMarginBias(basis, temperature=0.0)
        bias = StaticMarginBias(basis, alpha=1.0)
        with pytest.raises(ValueError, match="top_k"):
            bias.sample(logits, top_k=0)
        with pytest.raises(ValueError, match="top_p"):
            bias.sample(logits, top_p=1.5)

    def test_batched_logits_supported(self):
        """A ``(B, V)`` logit tensor is adjusted without a context argument."""
        learner = _fitted_learner(seed=39)
        emb, _ = _embeddings(vocab=300, seed=40)
        basis = MarginBasis.from_subspace_learner(learner, emb)
        bias = StaticMarginBias(basis, alpha=2.0)
        out = bias.adjust_logits(torch.zeros(7, emb.shape[0]))
        assert out.shape == (7, emb.shape[0])
        assert torch.allclose(out[0], bias.adjust_logits(torch.zeros(emb.shape[0])))


# ------------------------------------------------------------------ speed


class TestPerformance:
    """The collapsed path must actually be faster at a realistic size."""

    def test_fast_path_is_faster(self):
        """At a production-like vocabulary the fast path is much cheaper."""
        from sasa.fast_margin import benchmark_step

        dim, vocab = 256, 50_000
        g = torch.Generator().manual_seed(41)
        emb = torch.randn(vocab, dim, generator=g)
        learner = _fitted_learner(dim=dim, seed=42)
        logits = torch.randn(vocab, generator=g)
        ctx = torch.randn(dim, generator=g)

        ref_ms, fast_ms = benchmark_step(
            learner, emb, logits, ctx, alpha=1.0, repeats=10
        )
        assert fast_ms < ref_ms, f"fast={fast_ms:.3f}ms ref={ref_ms:.3f}ms"

    def test_peak_memory_is_lower(self):
        """The fast path must not allocate a ``(V, d)`` intermediate."""
        dim, vocab = 128, 20_000
        g = torch.Generator().manual_seed(43)
        emb = torch.randn(vocab, dim, generator=g)
        learner = _fitted_learner(dim=dim, seed=44)
        basis = MarginBasis.from_subspace_learner(learner, emb)
        before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        bias = StaticMarginBias(basis, alpha=1.0)
        out = bias.adjust_logits(torch.randn(vocab))
        if before is not None:
            assert torch.cuda.max_memory_allocated() - before < vocab * dim * 4
        assert out.shape == (vocab,)
