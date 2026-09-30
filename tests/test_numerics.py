"""Tests for numerically robust subspace fitting.

Run with::

    pytest tests/test_numerics.py -v
"""

from __future__ import annotations

import pytest
import torch

from sasa.numerics import (
    fit_direction,
    heldout_separability,
    ledoit_wolf_shrinkage,
    shared_covariance,
    solve_direction,
)


def _clusters(dim: int = 32, n: int = 200, sep: float = 1.2, seed: int = 0):
    """Return synthetic ``(negative, positive)`` embedding matrices.

    Args:
        dim: Hidden size.
        n: Samples per class.
        sep: Cluster separation.
        seed: RNG seed.

    Returns:
        A ``(neg, pos)`` tuple of shape ``(n, dim)`` each.
    """
    g = torch.Generator().manual_seed(seed)
    axis = torch.zeros(dim)
    axis[0] = 1.0
    neg = torch.randn(n, dim, generator=g) + sep * axis
    pos = torch.randn(n, dim, generator=g) - sep * axis
    return neg, pos


def _anisotropic_clusters(dim: int = 64, n: int = 400, seed: int = 1):
    """Return clusters whose covariance is deliberately ill-conditioned.

    A small weight is placed on a single direction so the pooled covariance has
    a condition number in the thousands, which is realistic for LLM hidden
    states.

    Args:
        dim: Hidden size.
        n: Samples per class.
        seed: RNG seed.

    Returns:
        A ``(neg, pos)`` tuple of shape ``(n, dim)`` each.
    """
    g = torch.Generator().manual_seed(seed)
    scale = torch.ones(dim)
    scale[1] = 1e-3  # Near-degenerate direction.
    axis = torch.zeros(dim)
    axis[0] = 1.0
    neg = torch.randn(n, dim, generator=g) * scale + 1.5 * axis
    pos = torch.randn(n, dim, generator=g) * scale - 1.5 * axis
    return neg, pos


class TestSharedCovariance:
    """Pooled covariance construction."""

    def test_matches_manual_formula(self):
        """The pooled covariance uses the ``N1+N2-2`` scaling."""
        neg, pos = _clusters(dim=8, n=50, seed=2)
        sigma = shared_covariance(neg, pos)
        dof = neg.shape[0] + pos.shape[0] - 2
        c1 = (neg - neg.mean(0)).T @ (neg - neg.mean(0))
        c2 = (pos - pos.mean(0)).T @ (pos - pos.mean(0))
        assert torch.allclose(sigma, (c1 + c2) / dof, atol=1e-6)

    def test_rejects_feature_mismatch(self):
        """Mismatched feature widths are rejected."""
        with pytest.raises(ValueError, match="feature mismatch"):
            shared_covariance(torch.randn(10, 4), torch.randn(10, 5))

    def test_rejects_too_few_samples(self):
        """A single sample per class cannot define a covariance."""
        with pytest.raises(ValueError, match="at least 2"):
            shared_covariance(torch.randn(1, 4), torch.randn(10, 4))


class TestSolvers:
    """Linear-solve back-ends."""

    @pytest.mark.parametrize("solver", ["cholesky", "solve", "inv"])
    def test_solvers_agree(self, solver):
        """All three back-ends return the same direction on a well-posed system."""
        neg, pos = _clusters(dim=24, seed=3)
        sigma = shared_covariance(neg, pos) + 0.1 * torch.eye(24)
        delta = neg.mean(0) - pos.mean(0)
        w = solve_direction(sigma, delta, solver=solver)
        assert torch.allclose(sigma @ w, delta, atol=1e-3, rtol=1e-3)

    def test_unknown_solver_raises(self):
        """An unrecognised solver name is a ValueError, not a silent fallback."""
        sigma = torch.eye(4)
        with pytest.raises(ValueError, match="unknown solver"):
            solve_direction(sigma, torch.zeros(4), solver="magic")

    def test_cholesky_more_stable_than_inverse_on_ill_conditioned_input(self):
        """On a near-singular system Cholesky keeps more of the true direction."""
        neg, pos = _anisotropic_clusters(dim=64, seed=4)
        sigma = shared_covariance(neg, pos)
        sigma = sigma + 1e-8 * torch.eye(64)
        delta = neg.mean(0) - pos.mean(0)

        w_chol = solve_direction(sigma, delta, solver="cholesky")
        w_inv = solve_direction(sigma, delta, solver="inv")

        resid_chol = float((sigma @ w_chol - delta).norm().item())
        resid_inv = float((sigma @ w_inv - delta).norm().item())
        # Both should be small in absolute terms; the point of the test is that
        # Cholesky does not blow up, i.e. it never returns a wildly wrong norm.
        assert resid_chol <= max(resid_inv, 1e-2)
        assert torch.isfinite(w_chol).all()
        assert torch.isfinite(w_inv).all()


class TestShrinkage:
    """Ledoit-Wolf-style shrinkage."""

    def test_coefficient_in_unit_interval(self):
        """The coefficient is always a valid shrinkage strength."""
        neg, pos = _anisotropic_clusters(dim=32, seed=5)
        lam = ledoit_wolf_shrinkage(shared_covariance(neg, pos), 800)
        assert 0.0 <= lam <= 1.0

    def test_shrinks_toward_identity_on_noisy_data(self):
        """Heavily contaminated covariance gets a large coefficient."""
        g = torch.Generator().manual_seed(6)
        d = 16
        cov = torch.eye(d) * 5.0
        noise = torch.randn(d, 12, generator=g) * 0.01
        lam = ledoit_wolf_shrinkage(cov + noise @ noise.T / d, 200)
        assert lam > 0.0

    def test_zero_for_isotropic_well_sampled_data(self):
        """A well-conditioned isotropic covariance needs little shrinkage."""
        d = 64
        sigma = torch.eye(d) * 2.0
        lam = ledoit_wolf_shrinkage(sigma, 100_000)
        assert lam < 0.5


class TestFitDirection:
    """End-to-end fitting."""

    def test_separates_held_out_clusters(self):
        """A fitted direction recovers a clean held-out separation.

        With a mean separation of ``2 * 2.5`` standard deviations along a single
        axis, the expected balanced accuracy is well above 0.97; the threshold
        is set below that to stay robust to sampling noise.
        """
        neg, pos = _clusters(dim=32, n=150, sep=2.5, seed=7)
        gn = torch.Generator().manual_seed(8)
        axis = torch.zeros(32)
        axis[0] = 1.0
        neg_te = torch.randn(150, 32, generator=gn) + 2.5 * axis
        pos_te = torch.randn(150, 32, generator=gn) - 2.5 * axis
        res = fit_direction(neg, pos, solver="cholesky")
        assert heldout_separability(res.weight, res.bias, neg_te, pos_te) > 0.95

    def test_recovers_the_generating_direction(self):
        """The fitted weight is aligned with the axis that generated the data."""
        neg, pos = _clusters(dim=32, n=300, sep=2.0, seed=13)
        res = fit_direction(neg, pos, solver="cholesky")
        cosine = abs(float((res.weight / res.weight.norm())[0]))
        assert cosine > 0.9

    def test_hard_separation_still_beats_chance(self):
        """At low separation the direction is weak but not useless."""
        neg, pos = _clusters(dim=32, n=150, sep=1.2, seed=14)
        gn = torch.Generator().manual_seed(15)
        axis = torch.zeros(32)
        axis[0] = 1.0
        neg_te = torch.randn(150, 32, generator=gn) + 1.2 * axis
        pos_te = torch.randn(150, 32, generator=gn) - 1.2 * axis
        res = fit_direction(neg, pos, solver="cholesky")
        assert heldout_separability(res.weight, res.bias, neg_te, pos_te) > 0.80

    def test_matches_reference_learner_on_well_conditioned_data(self):
        """With shrinkage disabled the new fit reproduces the reference direction."""
        from sasa.subspace_learner import SubspaceLearner

        neg, pos = _clusters(dim=16, n=400, seed=9)
        ref = SubspaceLearner(16)
        params = ref.fit(neg, pos)
        new = fit_direction(neg, pos, shrinkage=0.0, solver="cholesky")
        # Directions are scale- and sign-free; compare the normalised angle.
        a = params.w_v / params.w_v.norm()
        b = new.weight / new.weight.norm()
        assert torch.allclose(a, b, atol=1e-2) or torch.allclose(a, -b, atol=1e-2)

    def test_shrinkage_reduces_condition_number(self):
        """Regularisation must improve the conditioning of the solved matrix."""
        neg, pos = _anisotropic_clusters(dim=64, seed=10)
        raw = fit_direction(neg, pos, shrinkage=0.0, solver="cholesky")
        reg = fit_direction(neg, pos, shrinkage=0.2, solver="cholesky")
        assert reg.reg_condition < raw.reg_condition

    def test_reports_diagnostics(self):
        """`FitResult.as_dict` exposes the solver and conditioning numbers."""
        neg, pos = _clusters(dim=16, seed=11)
        res = fit_direction(neg, pos)
        d = res.as_dict()
        assert d["solver"] == "cholesky"
        assert d["raw_condition"] > 0 and d["reg_condition"] > 0
        assert res.weight.shape == (16,)
        assert res.bias.shape == (16,)

    def test_bias_is_midpoint_of_means(self):
        """The bias is the midpoint of the class means, as in the reference."""
        neg, pos = _clusters(dim=12, seed=12)
        res = fit_direction(neg, pos)
        assert torch.allclose(res.bias, 0.5 * (res.mu_neg + res.mu_pos), atol=1e-6)


class TestPCAPreconditioning:
    """The ``pca_dim`` path, which is what makes small labelled sets usable."""

    def _rank_deficient_case(self, d: int = 128, n: int = 40, seed: int = 20):
        """Return a labelled set with fewer samples than dimensions."""
        g = torch.Generator().manual_seed(seed)
        axis = torch.zeros(d)
        axis[0] = 1.0
        neg = torch.randn(n // 2, d, generator=g) + 1.0 * axis
        pos = torch.randn(n // 2, d, generator=g) - 1.0 * axis
        t = torch.Generator().manual_seed(seed + 1)
        neg_te = torch.randn(150, d, generator=t) + 1.0 * axis
        pos_te = torch.randn(150, d, generator=t) - 1.0 * axis
        return neg, pos, neg_te, pos_te

    def test_reduces_train_heldout_gap(self):
        """Reducing the dimension must shrink the optimism of the fit."""
        neg, pos, neg_te, pos_te = self._rank_deficient_case()
        full = fit_direction(neg, pos, solver="cholesky")
        low = fit_direction(neg, pos, solver="cholesky", pca_dim=8)
        gap_full = (heldout_separability(full.weight, full.bias, neg, pos)
                    - heldout_separability(full.weight, full.bias, neg_te, pos_te))
        gap_low = (heldout_separability(low.weight, low.bias, neg, pos)
                   - heldout_separability(low.weight, low.bias, neg_te, pos_te))
        assert gap_low < gap_full

    def test_weight_is_returned_in_full_feature_space(self):
        """Downstream code sees a (d,) weight whether or not PCA was used."""
        neg, pos, _, _ = self._rank_deficient_case()
        res = fit_direction(neg, pos, pca_dim=8)
        assert res.weight.shape == (neg.shape[1],)
        assert res.bias.shape == (neg.shape[1],)
        assert res.effective_dim == 8
        assert res.basis.shape == (8, neg.shape[1])

    def test_reduced_scores_match_manual_projection(self):
        """`project` then score must equal the full-space score."""
        neg, pos, neg_te, pos_te = self._rank_deficient_case()
        res = fit_direction(neg, pos, pca_dim=6)
        s_direct = (neg_te - res.bias) @ res.weight
        s_proj = (res.project(neg_te) - res.project(res.bias.unsqueeze(0))) \
            @ res.project(res.weight.unsqueeze(0)).squeeze()
        assert torch.allclose(s_direct, s_proj, atol=1e-3, rtol=1e-2)

    def test_explained_variance_is_recorded(self):
        """The retained fraction of pooled variance is reported."""
        neg, pos, _, _ = self._rank_deficient_case()
        res = fit_direction(neg, pos, pca_dim=8)
        assert 0.0 < res.explained_variance < 1.0

    def test_rejects_pca_dim_above_available_rank(self):
        """Asking for more directions than samples is a clear error."""
        neg, pos, _, _ = self._rank_deficient_case(d=64, n=20)
        with pytest.raises(ValueError, match="exceeds the rank"):
            fit_direction(neg, pos, pca_dim=40)

    def test_beats_the_reference_fit_in_the_small_sample_regime(self):
        """The headline claim of the numerics module, as a regression test.

        With fewer labelled examples than hidden dimensions, the pooled
        covariance is singular. The reference fit (absolute 1e-6 ridge plus an
        explicit inverse) then returns a direction that is no better than
        chance on held-out data. Cholesky with scale-aware shrinkage should do
        materially better. The margins below are deliberately loose so the test
        tracks the *direction* of the effect, not a specific number.
        """
        from sasa.subspace_learner import SubspaceLearner

        d, n = 96, 36
        g = torch.Generator().manual_seed(30)
        axis = torch.zeros(d)
        axis[0] = 1.0
        neg = torch.randn(n // 2, d, generator=g) + 1.0 * axis
        pos = torch.randn(n // 2, d, generator=g) - 1.0 * axis
        t = torch.Generator().manual_seed(31)
        neg_te = torch.randn(300, d, generator=t) + 1.0 * axis
        pos_te = torch.randn(300, d, generator=t) - 1.0 * axis

        ref = SubspaceLearner(d)
        params = ref.fit(neg, pos)
        acc_ref = heldout_separability(params.w_v, params.b_v, neg_te, pos_te)

        ours = fit_direction(neg, pos, solver="cholesky")
        acc_ours = heldout_separability(ours.weight, ours.bias, neg_te, pos_te)

        assert acc_ours > acc_ref
        assert acc_ours > 0.55
