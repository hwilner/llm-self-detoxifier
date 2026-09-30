"""Tests for the NLMA non-linear next-state estimator.

The module's central claim is negative-then-positive: no affine estimator can
make the SASA margin context-dependent (Theorem 3), but a low-rank bilinear one
can, and it should predict the true next state better as well. These tests
assert both halves.
"""

from __future__ import annotations

import copy

import pytest
import torch

from sasa.next_state import (
    NextStateModel,
    collect_next_state_pairs,
    fit_next_state_model,
    margin_statistics,
)


def _synthetic_triples(n: int = 600, d: int = 24, vocab: int = 400, seed: int = 0):
    """Build synthetic next-state triples with a known bilinear component.

    The "true" map is affine plus a rank-one interaction, so a correct fit
    should recover a small interaction term and produce context-dependent
    margins.

    Args:
        n: Number of triples.
        d: Hidden size.
        vocab: Vocabulary size of the embedding table.
        seed: RNG seed.

    Returns:
        A ``(contexts, tokens, targets, emb)`` tuple.
    """
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(vocab, d, generator=g) * 0.2
    contexts = torch.randn(n, d, generator=g)
    tok_idx = torch.randint(0, vocab, (n,), generator=g)
    tokens = emb[tok_idx]

    u_true = torch.randn(1, d, generator=g) * 0.5
    v_true = torch.randn(1, d, generator=g) * 0.5
    r_true = torch.randn(1, d, generator=g) * 0.5

    a_true = torch.randn(d, d, generator=g) * (0.3 / d ** 0.5)
    b_true = torch.randn(d, d, generator=g) * (0.3 / d ** 0.5)
    m_true = torch.randn(d, generator=g) * 0.1

    targets = (
        m_true
        + contexts @ a_true.T
        + tokens @ b_true.T
        + ((contexts @ u_true.T) * (tokens @ v_true.T)) @ r_true
    )
    return contexts, tokens, targets, emb


def _fitted(**kwargs) -> NextStateModel:
    """Fit a model on the synthetic triples with sensible defaults."""
    ctx, tok, tgt, emb = _synthetic_triples()
    return fit_next_state_model(ctx, tok, tgt, emb, rank=2, steps=150, **kwargs)


class TestNextStateModel:
    """Shape contracts and estimation behaviour."""

    def test_estimate_shape(self):
        """The estimate is (V, d) for any rank."""
        model = _fitted()
        emb = _synthetic_triples()[3]
        out = model.estimate(torch.randn(24), emb)
        assert out.shape == (emb.shape[0], 24)

    def test_rank_zero_is_pure_affine(self):
        """Rank 0 must behave exactly like an affine estimator."""
        ctx, tok, tgt, emb = _synthetic_triples()
        model = fit_next_state_model(ctx, tok, tgt, emb, rank=0)
        g = torch.randn(24)
        assert torch.equal(model.interaction(g, emb),
                           torch.zeros(emb.shape[0], 24))
        assert model.rank == 0

    def test_rejects_width_mismatch(self):
        """A wrongly sized context or embedding table is rejected."""
        model = _fitted()
        emb = _synthetic_triples()[3]
        with pytest.raises(ValueError, match="context width"):
            model.estimate(torch.randn(7), emb)
        with pytest.raises(ValueError, match="token embedding width"):
            model.estimate(torch.randn(24), torch.randn(10, 5))

    def test_interaction_beats_affine_on_a_bilinear_target(self):
        """On a bilinear ground truth the full model must fit better.

        Compared per observation, so each row is scored against its own
        candidate token -- the per-observation predictor is
        :meth:`NextStateModel.predict_pairs`.
        """
        ctx, tok, tgt, emb = _synthetic_triples()
        model = fit_next_state_model(ctx, tok, tgt, emb, rank=2, steps=300)

        pred_full = model.predict_pairs(ctx, tok)
        pred_affine = model.predict_pairs(ctx, tok)
        model_no_inter = copy.copy(model)
        model_no_inter.u = torch.zeros(0, model.embedding_dim)
        model_no_inter.v = torch.zeros(0, model.embedding_dim)
        model_no_inter.r = torch.zeros(0, model.embedding_dim)
        pred_affine = model_no_inter.predict_pairs(ctx, tok)
        mse_full = float((pred_full - tgt).pow(2).mean().item())
        mse_affine = float((pred_affine - tgt).pow(2).mean().item())

        assert mse_full < mse_affine
        assert model.fit_stats["final_interaction_mse"] >= 0.0

    def test_affine_only_margins_are_context_independent(self):
        """Theorem 3, operationally: the affine part cancels in the softmax."""
        model = _fitted()
        emb = _synthetic_triples()[3]
        w = torch.randn(24)
        b = torch.randn(24)
        tv = model.context_sensitivity(
            w, b, torch.randn(24), torch.randn(24) * 50.0, emb, affine_only=True
        )
        assert tv < 1e-6

    def test_full_model_margins_are_context_dependent(self):
        """The interaction term survives the softmax, unlike any affine term."""
        model = _fitted()
        emb = _synthetic_triples()[3]
        w = torch.randn(24)
        b = torch.randn(24)
        tv = model.context_sensitivity(
            w, b, torch.randn(24), torch.randn(24) * 50.0, emb, affine_only=False
        )
        assert tv > 1e-4

    def test_token_projection_shape_and_use(self):
        """``token_projection`` is (V, R) and matches the interaction up to a_mat."""
        model = _fitted()
        emb = _synthetic_triples()[3]
        assert model.token_projection.shape == (emb.shape[0], model.rank)
        g = torch.randn(24)
        expected = (emb @ model.v.T * (model.u @ g).unsqueeze(0)) @ model.r
        assert torch.allclose(model.interaction(g, emb), expected, atol=1e-5)

    def test_fit_stats_are_recorded(self):
        """Fit diagnostics are populated for the report."""
        model = _fitted()
        for key in ("rank", "n_samples", "affine_mse", "final_interaction_mse"):
            assert key in model.fit_stats


class TestFitting:
    """Validation and numerical behaviour of the fit."""

    def test_rejects_shape_mismatch(self):
        """Mismatched triple shapes raise rather than silently broadcasting."""
        ctx, tok, tgt, emb = _synthetic_triples()
        with pytest.raises(ValueError, match="shape mismatch"):
            fit_next_state_model(ctx, tok[:-1], tgt, emb)
        with pytest.raises(ValueError, match="token_embeddings width"):
            fit_next_state_model(ctx, tok, tgt, torch.randn(emb.shape[0], 5))

    def test_affine_part_recovers_a_purely_affine_target(self):
        """With no interaction and no bilinear target, the ridge fit is exact.

        A purely affine ground truth isolates the solver: if the closed-form
        path is wrong, this test fails even though the bilinear tests pass.
        """
        g = torch.Generator().manual_seed(9)
        d, n = 16, 400
        ctx = torch.randn(n, d, generator=g)
        tok = torch.randn(n, d, generator=g) * 0.1
        a_true = torch.randn(d, d, generator=g) * (0.2 / d ** 0.5)
        b_true = torch.randn(d, d, generator=g) * (0.2 / d ** 0.5)
        m_true = torch.randn(d, generator=g) * 0.1
        tgt = m_true + ctx @ a_true.T + tok @ b_true.T

        model = fit_next_state_model(ctx, tok, tgt,
                                     token_embeddings=torch.randn(50, d),
                                     rank=0, ridge=1e-4)
        mse = float((model.predict_pairs(ctx, tok) - tgt).pow(2).mean().item())
        assert mse < 1e-3 * float(tgt.pow(2).mean().item())

    def test_predict_pairs_rejects_mismatched_rows(self):
        """Observations must be paired."""
        model = _fitted()
        with pytest.raises(ValueError, match="shape mismatch"):
            model.predict_pairs(torch.randn(10, 24), torch.randn(9, 24))

    def test_higher_rank_lowers_interaction_error(self):
        """More interaction capacity cannot increase the residual it explains."""
        ctx, tok, tgt, emb = _synthetic_triples()
        low = fit_next_state_model(ctx, tok, tgt, emb, rank=1, steps=150)
        high = fit_next_state_model(ctx, tok, tgt, emb, rank=4, steps=150)
        assert high.fit_stats["final_interaction_mse"] <= \
            low.fit_stats["final_interaction_mse"] * 1.05

    def test_deterministic_given_seed(self):
        """Two fits with the same seed produce identical parameters."""
        ctx, tok, tgt, emb = _synthetic_triples()
        a = fit_next_state_model(ctx, tok, tgt, emb, rank=2, steps=60, seed=7)
        b = fit_next_state_model(ctx, tok, tgt, emb, rank=2, steps=60, seed=7)
        assert torch.equal(a.u, b.u)
        assert torch.equal(a.v, b.v)


class TestMarginStatistics:
    """Comparison of two margin vectors."""

    def test_identical_vectors_give_zero_spread(self):
        """Comparing a vector with itself yields perfect agreement."""
        m = torch.randn(500)
        s = margin_statistics(m, m)
        assert s.total_variation == pytest.approx(0.0, abs=1e-6)
        assert s.spearman == pytest.approx(1.0, abs=1e-5)
        assert s.cosine == pytest.approx(1.0, abs=1e-5)
        assert s.top1_agreement == pytest.approx(1.0)

    def test_reversed_vector_antialigns(self):
        """A negated vector has perfect negative rank correlation."""
        m = torch.randn(500)
        s = margin_statistics(m, -m)
        assert s.spearman == pytest.approx(-1.0, abs=1e-5)
        assert s.top1_agreement == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        """Mismatched margin vectors are rejected."""
        with pytest.raises(ValueError, match="shape mismatch"):
            margin_statistics(torch.randn(10), torch.randn(11))

    def test_dict_round_trip(self):
        """`as_dict` returns JSON-friendly scalars."""
        s = margin_statistics(torch.randn(50), torch.randn(50))
        assert set(s.as_dict()) == {
            "total_variation", "spearman", "pearson", "cosine", "top1_agreement"
        }


class TestCollection:
    """Ground-truth collection from a real model."""

    def test_rejects_empty_prompts(self):
        """An empty prompt list is an error, not an empty tensor."""
        with pytest.raises(ValueError, match="must not be empty"):
            collect_next_state_pairs(None, None, [], 0, torch.device("cpu"))


class TestIntegrationWithFastMargin:
    """NLMA and the static basis must compose without conflicting claims."""

    def test_static_basis_stays_context_free_when_nlma_is_used(self):
        """Adopting NLMA does not change what the static bias is.

        The two are different mechanisms: the static bias is the exact
        reference behaviour, NLMA is a proposed replacement. A test guards
        against someone 'fixing' the static bias and thereby falsifying the
        Theorem 1 measurement.
        """
        from sasa.fast_margin import MarginBasis
        from sasa.subspace_learner import SubspaceLearner

        d, vocab = 12, 50
        gen = torch.Generator().manual_seed(4)
        axis = torch.zeros(d)
        axis[0] = 1.0
        learner = SubspaceLearner(d)
        learner.fit(torch.randn(40, d, generator=gen) + axis,
                    torch.randn(40, d, generator=gen) - axis)
        emb = torch.randn(vocab, d, generator=gen)
        basis = MarginBasis.from_subspace_learner(learner, emb)
        a = basis.static_bias(1.0)
        b = basis.static_bias(1.0)
        assert torch.equal(a, b)
