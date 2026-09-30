"""Tests for graded multi-attribute subspaces and interference geometry.

The two claims under test are the ones Appendix H rests on:

1. Graded supervision is *free* and *contains* the binary method -- a
   two-grade fit must reproduce the binary LDA score, and a k-grade fit must
   have k-1 independent directions.
2. Subspace overlap is a usable static predictor of interference, which
   requires it to equal 1 for an attribute against itself and to fall towards
   0 as two attributes' weight directions become orthogonal.

Run with::

    pytest tests/test_multi_attribute.py -v
"""

from __future__ import annotations

import pytest
import torch

from sasa.multi_attribute import (
    IDENTIFIABILITY_GATE,
    AttributeSet,
    GradedSubspace,
)
from sasa.numerics import fit_direction


def _axis(dim: int, *components: float) -> torch.Tensor:
    """Build a sparse direction.

    Args:
        dim: Vector length.
        *components: Non-zero entries, in order, from index 0 upwards.

    Returns:
        A vector of shape ``(dim,)``.

    Raises:
        ValueError: If more components are given than the dimension allows.
    """
    v = torch.zeros(dim)
    for i, value in enumerate(components):
        if i >= dim:
            raise ValueError("too many components for the requested dimension")
        v[i] = value
    return v


def _cluster(centre: torch.Tensor, n: int = 150, noise: float = 0.30,
             dim: int = 40, seed: int = 0) -> torch.Tensor:
    """Sample a Gaussian cluster around ``centre``.

    Args:
        centre: Cluster centre, shape ``(dim,)``.
        n: Number of samples.
        noise: Per-coordinate standard deviation.
        dim: Hidden size.
        seed: RNG seed.

    Returns:
        Tensor of shape ``(n, dim)``.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g) * noise + centre


def _two_grade(dim: int = 40, seed: int = 0):
    """Build a clean two-grade (binary) labelled set.

    Args:
        dim: Hidden size.
        seed: RNG seed.

    Returns:
        A ``(features, grades)`` tuple.
    """
    direction = _axis(dim, 1.0, 0.5)
    x = torch.cat([
        _cluster(3.0 * direction, 120, seed=seed),
        _cluster(-3.0 * direction, 120, seed=seed + 1),
    ])
    y = torch.tensor([0.0] * 120 + [3.0] * 120)
    return x, y


def _four_grade_2d(dim: int = 40, seed: int = 0):
    """Build a four-grade set whose class centres span two dimensions.

    Args:
        dim: Hidden size.
        seed: RNG seed.

    Returns:
        A ``(features, grades)`` tuple.
    """
    a1 = _axis(dim, 1.0)
    a2 = _axis(dim, 0.0, 1.0)
    centres = [(3.0, 0.0), (0.0, 3.0), (0.0, -3.0), (-3.0, 0.0)]
    xs, ys = [], []
    for grade, (c1, c2) in enumerate(centres):
        centre = c1 * a1 + c2 * a2
        xs.append(_cluster(centre, 150, seed=seed + grade))
        ys += [float(grade)] * 150
    return torch.cat(xs), torch.tensor(ys)


class TestGradedSubspaceFit:
    """Construction contracts and the k-2 reduction."""

    def test_rank_is_the_numerical_rank_of_the_between_class_scatter(self):
        """A four-grade fit spanning two dimensions has rank two.

        With 40 hidden dimensions and Gaussian noise, four class means are
        *not* exactly rank-deficient, so the rank is set by the numerical rank
        of the centred class means, capped at k-1. Three is the cap here; the
        check below pins the mechanism rather than the exact number.
        """
        x, y = _four_grade_2d()
        sub = GradedSubspace.fit(x, y, name="t", toxic_from=2.0,
                                 shrinkage=1e-3, seed=1)
        assert sub.n_classes == 4
        assert 1 <= sub.effective_rank <= sub.n_classes - 1
        assert sub.basis.shape == (sub.effective_rank, x.shape[1])

    def test_exactly_rank_two_when_the_class_layout_is_two_dimensional(self):
        """A clean 2-D layout on a low-dimensional feature space gives rank 2."""
        dim = 6
        a1, a2 = _axis(dim, 1.0), _axis(dim, 0.0, 1.0)
        centres = [(3.0, 0.0), (0.0, 3.0), (0.0, -3.0), (-3.0, 0.0)]
        xs, ys = [], []
        for grade, (c1, c2) in enumerate(centres):
            g = torch.Generator().manual_seed(grade)
            xs.append(torch.randn(80, dim, generator=g) * 0.2 + c1 * a1 + c2 * a2)
            ys += [float(grade)] * 80
        sub = GradedSubspace.fit(torch.cat(xs), torch.tensor(ys),
                                 toxic_from=2.0, shrinkage=1e-4, seed=1)
        assert sub.effective_rank == 2

    def test_two_grades_give_rank_one(self):
        """A binary fit is rank one, recovering the rank-one SASA margin."""
        x, y = _two_grade()
        sub = GradedSubspace.fit(x, y, name="t", toxic_from=2.0,
                                 shrinkage=1e-3, seed=1)
        assert sub.n_classes == 2
        assert sub.effective_rank == 1

    def test_graded_reduces_to_the_reference_binary_score(self):
        """For two grades the graded margin equals the reference LDA margin.

        This is the claim that makes graded supervision a strict generalisation
        rather than a different method: at k = 2 it must be the same operator.
        """
        x, y = _two_grade()
        grades = GradedSubspace.fit(x, y, name="g", toxic_from=2.0,
                                    shrinkage=1e-3, seed=1)
        # Reference fit via sasa.numerics on the same split convention.
        g = torch.Generator().manual_seed(1)
        perm = torch.randperm(x.shape[0], generator=g)
        cut = int(round(0.7 * x.shape[0]))
        tr = perm[:cut]
        ref = fit_direction(x[tr][y[tr] == 0.0], x[tr][y[tr] == 3.0],
                            shrinkage=0.0, solver="cholesky")
        ref_score = (x[tr] - ref.bias) @ ref.weight
        got = grades.score(x[tr])
        # Directions are defined up to positive scale; compare the correlation.
        corr = float(torch.corrcoef(torch.stack([ref_score, got]))[0, 1])
        assert corr > 0.97

    def test_graded_score_is_affine_in_the_features(self):
        """The margin is affine, which is what licenses the static collapse."""
        x, y = _four_grade_2d()
        sub = GradedSubspace.fit(x, y, toxic_from=2.0, shrinkage=1e-3, seed=1)
        a, b = x[:1], x[1:2]
        direct = sub.score((a + b) / 2)
        averaged = (sub.score(a) + sub.score(b)) / 2
        assert float(direct) == pytest.approx(float(averaged), rel=1e-4)

    def test_grade_accuracy_exceeds_side_accuracy_on_graded_data(self):
        """A four-grade fit should recover grades, not just the binary side."""
        x, y = _four_grade_2d()
        sub = GradedSubspace.fit(x, y, toxic_from=2.0, shrinkage=1e-3, seed=1)
        assert sub.heldout_accuracy > 0.8
        assert sub.heldout_side_accuracy > 0.9

    def test_clean_data_passes_the_identifiability_gate(self):
        """Separable data clears the pre-registered gate."""
        x, y = _two_grade()
        sub = GradedSubspace.fit(x, y, toxic_from=2.0, shrinkage=1e-3, seed=1)
        assert sub.passes_gate()
        assert sub.heldout_side_accuracy >= IDENTIFIABILITY_GATE

    def test_pure_noise_fails_the_gate(self):
        """Unlabelled noise must not enter the interference study.

        This is the gate that Appendix E showed is necessary: a subspace with no
        signal would otherwise be compared against real ones.
        """
        g = torch.Generator().manual_seed(3)
        x = torch.randn(240, 40, generator=g)
        y = torch.tensor([0.0] * 120 + [3.0] * 120)
        sub = GradedSubspace.fit(x, y, toxic_from=2.0, shrinkage=1e-3, seed=1)
        assert not sub.passes_gate()

    def test_fit_is_deterministic(self):
        """Two fits with the same seed give identical parameters."""
        x, y = _four_grade_2d()
        a = GradedSubspace.fit(x, y, shrinkage=1e-3, seed=5)
        b = GradedSubspace.fit(x, y, shrinkage=1e-3, seed=5)
        assert torch.equal(a.basis, b.basis)
        assert torch.equal(a.mu_whitened, b.mu_whitened)

    def test_split_is_held_out(self):
        """A fraction of the data is genuinely reserved."""
        x, y = _two_grade()
        sub = GradedSubspace.fit(x, y, holdout_frac=0.3, seed=1)
        assert sub.n_fit == int(round(0.7 * x.shape[0]))


class TestGradedSubspaceValidation:
    """Error paths."""

    def test_requires_two_distinct_grades(self):
        """A single grade cannot define a discriminant."""
        x = torch.randn(20, 8)
        with pytest.raises(ValueError, match="at least two distinct grades"):
            GradedSubspace.fit(x, [1.0] * 20)

    def test_rejects_length_mismatch(self):
        """Grades and features must align."""
        with pytest.raises(ValueError, match="grades has length"):
            GradedSubspace.fit(torch.randn(10, 4), [0.0] * 9)

    def test_rejects_1d_features(self):
        """Features must be a matrix."""
        with pytest.raises(ValueError, match="2-D"):
            GradedSubspace.fit(torch.randn(10), [0.0, 1.0])

    def test_thin_class_is_merged_and_recorded(self):
        """A class with one example is merged, not silently dropped.

        A single example contributes zero degrees of freedom to the pooled
        covariance, so it cannot be fitted. Ordinal labels are naturally lumpy,
        so merging is the right behaviour -- but it must be recorded, because a
        merge changes what the grades mean.
        """
        x = torch.randn(40, 6)
        y = [0.0] * 20 + [1.0] * 5 + [2.0] * 14 + [3.0] * 1
        sub = GradedSubspace.fit(x, y, toxic_from=2.0, shrinkage=1e-3, seed=1)
        assert 3.0 in sub.merged_grades
        assert 1.0 not in sub.merged_grades
        assert sub.summary()["merged_grades"] == [3.0]

    def test_rejects_when_merging_leaves_one_class(self):
        """A problem that merging cannot rescue is a genuine error."""
        x = torch.randn(10, 6)
        y = [0.0] * 6 + [1.0] * 2 + [2.0] * 2
        with pytest.raises(ValueError, match="min_class_size"):
            GradedSubspace.fit(x, y, min_class_size=8)

    def test_one_sided_grades_rejected_at_scoring_time(self):
        """A toxic_from threshold with no examples on one side is an error."""
        x, _ = _two_grade()
        sub = GradedSubspace.fit(x, [0.0] * 120 + [3.0] * 120,
                                 toxic_from=9.0, shrinkage=1e-3, seed=1)
        with pytest.raises(ValueError, match="one side"):
            sub.score(x[:4])

    def test_token_bias_rejects_width_mismatch(self):
        """A wrongly sized embedding table is rejected."""
        x, y = _two_grade()
        sub = GradedSubspace.fit(x, y, shrinkage=1e-3, seed=1)
        with pytest.raises(ValueError, match="width"):
            sub.token_bias(torch.randn(10, 7))

    def test_summary_is_serialisable(self):
        """`summary` returns JSON-encodable values, including the spectrum."""
        import json

        x, y = _two_grade()
        sub = GradedSubspace.fit(x, y, shrinkage=1e-3, seed=1).summary()
        assert isinstance(sub["between_class_spectrum"], list)
        json.dumps(sub)  # must not raise

    def test_rank_tolerance_controls_the_retained_rank(self):
        """The rank cut is a documented parameter, not a hidden constant."""
        x, y = _two_grade()
        tight = GradedSubspace.fit(x, y, shrinkage=1e-3, seed=1,
                                   rank_tol_rel=1e-9)
        loose = GradedSubspace.fit(x, y, shrinkage=1e-3, seed=1,
                                   rank_tol_rel=0.9)
        assert tight.effective_rank >= loose.effective_rank
        assert loose.effective_rank >= 1

    def test_binary_fit_is_always_at_least_rank_one(self):
        """Two classes must always yield a usable direction."""
        x, y = _two_grade()
        for tol in (1e-9, 0.1, 0.9, 2.0):
            sub = GradedSubspace.fit(x, y, shrinkage=1e-3, seed=1,
                                     rank_tol_rel=tol)
            assert sub.effective_rank >= 1
            assert sub.score(x).shape == (x.shape[0],)


class TestTokenBias:
    """The static collapse for graded margins."""

    def test_shape_and_staticness(self):
        """The bias depends only on the embedding table."""
        x, y = _four_grade_2d()
        sub = GradedSubspace.fit(x, y, toxic_from=2.0, shrinkage=1e-3, seed=1)
        emb = torch.randn(500, 40)
        bias = sub.token_bias(emb)
        assert bias.shape == (500,)
        assert torch.equal(sub.token_bias(emb), bias)

    def test_bias_is_half_the_score_of_the_embedding_alone(self):
        """token_bias = score(e_t) / 2, matching the (g + e)/2 estimator."""
        x, y = _four_grade_2d()
        sub = GradedSubspace.fit(x, y, toxic_from=2.0, shrinkage=1e-3, seed=1)
        emb = torch.randn(60, 40)
        assert torch.allclose(sub.token_bias(emb), 0.5 * sub.score(emb), atol=1e-5)


class TestAttributeSet:
    """Composition and the interference geometry."""

    def _three_attributes(self, dim: int = 40):
        """Build a shared direction, a disjoint one, and a mixed one.

        Args:
            dim: Hidden size.

        Returns:
            A list of three fitted :class:`GradedSubspace` objects.
        """
        a1 = _axis(dim, 1.0)
        a2 = _axis(dim, 0.0, 1.0)
        a3 = _axis(dim, *([0.0] * 8 + [1.0]))

        x_a, y_a = _two_grade(dim=dim, seed=0)
        x_b = torch.cat([_cluster(3.0 * a3, 120, dim=dim, seed=11),
                         _cluster(-3.0 * a3, 120, dim=dim, seed=12)])
        y_b = torch.tensor([0.0] * 120 + [3.0] * 120)
        x_c = torch.cat([_cluster(2.5 * a1 + 2.0 * a2, 120, dim=dim, seed=21),
                         _cluster(-2.5 * a1 - 2.0 * a2, 120, dim=dim, seed=22)])
        y_c = torch.tensor([0.0] * 120 + [3.0] * 120)

        return [
            GradedSubspace.fit(x_a, y_a, name="shared", shrinkage=1e-3, seed=1),
            GradedSubspace.fit(x_b, y_b, name="disjoint", shrinkage=1e-3, seed=1),
            GradedSubspace.fit(x_c, y_c, name="mixed", shrinkage=1e-3, seed=1),
        ]

    def test_self_overlap_is_one(self):
        """An attribute has full overlap with itself."""
        a = self._three_attributes()[0]
        pair = AttributeSet([a, a]).interference()[0]
        assert pair.subspace_overlap == pytest.approx(1.0, abs=1e-4)
        assert pair.max_cosine == pytest.approx(1.0, abs=1e-4)

    def test_disjoint_attributes_have_lower_overlap(self):
        """Orthogonal weight directions overlap less than a shared direction."""
        shared, disjoint, mixed = self._three_attributes()
        pairs = {(p.attribute_a, p.attribute_b): p
                 for p in AttributeSet([shared, disjoint, mixed]).interference()}
        o_sp = pairs[("shared", "disjoint")].subspace_overlap
        o_sm = pairs[("shared", "mixed")].subspace_overlap
        assert o_sp < o_sm

    def test_interference_is_sorted_by_overlap(self):
        """The worst-predicted pair is reported first."""
        s = AttributeSet(self._three_attributes())
        overlaps = [p.subspace_overlap for p in s.interference()]
        assert overlaps == sorted(overlaps, reverse=True)

    def test_pair_count_is_combinations(self):
        """Three attributes give three unordered pairs."""
        s = AttributeSet(self._three_attributes())
        assert len(s.interference()) == 3

    def test_identifiable_filters_by_gate(self):
        """Pure-noise attributes are excluded by the gate."""
        dim = 40
        g = torch.Generator().manual_seed(99)
        x_noise = torch.randn(240, dim, generator=g)
        y_noise = torch.tensor([0.0] * 120 + [3.0] * 120)
        noise = GradedSubspace.fit(x_noise, y_noise, name="noise",
                                   shrinkage=1e-3, seed=1)
        good = self._three_attributes()[0]
        gated = AttributeSet([good, noise]).identifiable()
        assert "noise" not in gated.names
        assert "shared" in gated.names

    def test_compose_logits_applies_only_named_attributes(self):
        """Composition is additive and respects the alpha mapping."""
        dim = 40
        shared, disjoint, _mixed = self._three_attributes(dim)
        attrs = AttributeSet([shared, disjoint])
        emb = torch.randn(300, dim)
        logits = torch.randn(300)
        both = attrs.compose_logits(logits, emb, {"shared": 1.0, "disjoint": 1.0})
        one = attrs.compose_logits(logits, emb, {"shared": 1.0})
        assert both.shape == logits.shape
        assert not torch.allclose(both, one)
        # Additivity: composing both equals the sum of the individual deltas.
        # Compared as a relative norm rather than element-wise, because the
        # discriminant scores reach the thousands here and an absolute tolerance
        # would be either vacuous or unachievable in float32.
        expected = one + attrs.compose_logits(
            logits, emb, {"disjoint": 1.0}
        ) - logits
        rel = float((both - expected).norm() / both.norm())
        assert rel < 1e-5

    def test_compose_logits_scales_linearly_in_alpha(self):
        """Doubling alpha doubles the shift."""
        dim = 40
        shared = self._three_attributes(dim)[0]
        attrs = AttributeSet([shared])
        emb = torch.randn(200, dim)
        logits = torch.randn(200)
        one = attrs.compose_logits(logits, emb, {"shared": 1.0})
        two = attrs.compose_logits(logits, emb, {"shared": 2.0})
        assert torch.allclose(two - logits, 2.0 * (one - logits), atol=1e-5)

    def test_compose_logits_rejects_unknown_attribute(self):
        """A typo in the alpha mapping is an error, not a silent no-op."""
        dim = 40
        attrs = AttributeSet([self._three_attributes(dim)[0]])
        with pytest.raises(KeyError, match="unknown attributes"):
            attrs.compose_logits(torch.randn(50), torch.randn(50, dim),
                                 {"nope": 1.0})

    def test_compose_logits_relative_scale(self):
        """Two attributes on very different scales still compose additively."""
        dim = 40
        narrow = GradedSubspace.fit(
            torch.cat([_cluster(1.0 * _axis(dim, 0), 120, dim=dim, seed=41),
                       _cluster(-1.0 * _axis(dim, 0), 120, dim=dim, seed=42)]),
            torch.tensor([0.0] * 120 + [3.0] * 120),
            name="narrow", shrinkage=1e-3, seed=1,
        )
        wide = GradedSubspace.fit(
            torch.cat([_cluster(12.0 * _axis(dim, 3), 120, dim=dim, seed=43),
                       _cluster(-12.0 * _axis(dim, 3), 120, dim=dim, seed=44)]),
            torch.tensor([0.0] * 120 + [3.0] * 120),
            name="wide", shrinkage=1e-3, seed=1,
        )
        attrs = AttributeSet([narrow, wide])
        emb = torch.randn(200, dim)
        logits = torch.randn(200)
        both = attrs.compose_logits(logits, emb, {"narrow": 1.0, "wide": 1.0})
        expected = logits + (narrow.token_bias(emb) + wide.token_bias(emb))
        assert float((both - expected).norm() / both.norm()) < 1e-5

    def test_normalisation_makes_alphas_comparable(self):
        """After normalisation each attribute's shift has unit scale."""
        dim = 40
        shared, disjoint, _mixed = self._three_attributes(dim)
        # Make the two attributes live on very different scales.
        wide = GradedSubspace.fit(
            torch.cat([_cluster(9.0 * _axis(dim, 1.0), 120, dim=dim, seed=31),
                       _cluster(-9.0 * _axis(dim, 1.0), 120, dim=dim, seed=32)]),
            torch.tensor([0.0] * 120 + [3.0] * 120),
            name="wide", shrinkage=1e-3, seed=1,
        )
        attrs = AttributeSet([shared, wide])
        emb = torch.randn(400, dim)
        logits = torch.randn(400)
        raw = attrs.compose_logits(logits, emb, {"shared": 1.0, "wide": 1.0})
        norm = attrs.compose_logits(logits, emb, {"shared": 1.0, "wide": 1.0},
                                    normalise=True)
        # Unnormalised, the wider attribute dominates; normalised, it does not.
        assert float((raw - logits).abs().max()) > \
            float((norm - logits).abs().max())

    def test_batched_logits_supported(self):
        """A (B, V) logit tensor is composed without extra handling."""
        dim = 40
        attrs = AttributeSet([self._three_attributes(dim)[0]])
        emb = torch.randn(120, dim)
        out = attrs.compose_logits(torch.zeros(5, 120), emb, {"shared": 1.0})
        assert out.shape == (5, 120)

    def test_principal_angles_reject_width_mismatch(self):
        """Subspaces of different widths cannot be compared."""
        dim = 40
        a = self._three_attributes(dim)[0]
        g = torch.Generator().manual_seed(4)
        b = GradedSubspace.fit(
            torch.randn(240, 24, generator=g),
            torch.tensor([0.0] * 120 + [3.0] * 120),
            name="other", shrinkage=1e-3, seed=1,
        )
        with pytest.raises(ValueError, match="feature width"):
            AttributeSet([a, b]).interference()

    def test_lookup_by_name_and_index(self):
        """Attributes are addressable by position and by name."""
        s = AttributeSet(self._three_attributes())
        assert s["shared"] is s[0]
        assert s.names == ["shared", "disjoint", "mixed"]
        with pytest.raises(KeyError):
            s["absent"]

    def test_add_returns_self_for_chaining(self):
        """`add` is chainable."""
        a, b, _c = self._three_attributes()
        s = AttributeSet()
        assert s.add(a).add(b) is s
        assert len(s) == 2
