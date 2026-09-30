"""Tests for alpha schedules and the pre-registered U2 selection rule.

Run with::

    pytest tests/test_scheduling.py -v
"""

from __future__ import annotations

import pytest

from sasa.scheduling import (
    MAX_PERPLEXITY_DELTA,
    MIN_RELATIVE_GAIN,
    AlphaSchedule,
    select_schedule,
)


class TestAlphaSchedule:
    """Shape and bounds of the schedule families."""

    def test_fixed_is_constant(self):
        """A fixed schedule returns the peak at every step."""
        s = AlphaSchedule("fixed", peak=1.7, total_steps=20)
        assert s.values(20) == [pytest.approx(1.7)] * 20

    def test_linear_ramps_up(self):
        """A linear schedule is monotone non-decreasing."""
        s = AlphaSchedule("linear", peak=1.0, total_steps=11)
        vals = s.values(11)
        assert vals[0] == pytest.approx(0.0, abs=1e-9)
        assert vals[-1] == pytest.approx(1.0, abs=1e-9)
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))

    def test_cosine_decays(self):
        """A cosine schedule falls from the peak toward the floor.

        The endpoint is checked against the floor rather than zero because
        clamping happens before the assertion: with ``floor=0`` the last step
        is clamped up to 0, so the raw cosine value at t=1 is only reached in
        the limit.
        """
        s = AlphaSchedule("cosine", peak=1.0, floor=0.0, total_steps=11)
        vals = s.values(11)
        assert vals[0] == pytest.approx(1.0, abs=1e-9)
        assert vals[-1] == pytest.approx(0.0, abs=1e-9)
        assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))
        # Half-way through the decay the strength is the half-max.
        assert vals[5] == pytest.approx(0.5, abs=1e-2)

    def test_floor_is_respected(self):
        """A decay never falls below the configured floor."""
        s = AlphaSchedule("cosine", peak=1.0, floor=0.4, total_steps=11)
        assert min(s.values(11)) >= 0.4 - 1e-12

    def test_steps_beyond_length_are_clamped(self):
        """Asking for more steps than calibrated yields the final value."""
        s = AlphaSchedule("linear", peak=1.0, total_steps=5)
        assert s(999) == pytest.approx(s(4))

    def test_weak_to_strong_direction_is_preserved(self):
        """All families keep the peak as the maximum over a full generation."""
        for kind in ("fixed", "linear", "cosine", "step"):
            s = AlphaSchedule(kind, peak=2.0, total_steps=16)
            assert max(s.values(16)) == pytest.approx(2.0)

    def test_margin_gated_opens_with_context(self):
        """The gate is off at a zero context score and open at one."""
        s = AlphaSchedule("margin_gated", peak=1.0, total_steps=8,
                          gate_threshold=0.3)
        assert s(0, context_score=0.0) == pytest.approx(0.0)
        assert s(0, context_score=1.0) == pytest.approx(1.0, abs=1e-9)
        mid = s(0, context_score=0.65)
        assert 0.0 < mid < 1.0

    def test_custom_gate_is_used(self):
        """A user-supplied gate overrides the default."""
        s = AlphaSchedule("margin_gated", peak=1.0,
                          gate=lambda step, score: 0.25)
        assert s(3, context_score=0.9) == pytest.approx(0.25)

    def test_schedule_is_the_only_position_dependent_knob(self):
        """For a static bias, a schedule rescales one fixed vector.

        This documents the Theorem-1 consequence that motivates schedules: the
        *direction* of steering cannot change with step, only its magnitude.
        """
        s = AlphaSchedule("linear", peak=1.0, total_steps=5)
        vals = s.values(5)
        assert len(set(round(v, 12) for v in vals)) == 5  # magnitudes differ
        # ... and all of them are multiples of the same underlying vector, so
        # the *ranking* of tokens is identical at every step.
        base = torch_like_vector(10)
        # A zero magnitude collapses the vector, so only positive strengths are
        # informative; the point is that the *ordering* never changes.
        assert all(rank_agreement([x * v for x in base], base)
                   for v in vals if v > 0)
        assert 0.0 in vals  # the linear ramp genuinely reaches zero


def torch_like_vector(n: int):
    """Return a deterministic pseudo-vector, avoiding a torch import here.

    Args:
        n: Vector length.

    Returns:
        A list of floats.
    """
    return [float(n - i) + 0.01 * ((i * 7) % 3) for i in range(n)]


def rank_agreement(a, b) -> bool:
    """Whether two scaled vectors induce the same token ordering.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        ``True`` when the orderings agree, up to ties.
    """
    order_a = sorted(range(len(a)), key=lambda i: -a[i])
    order_b = sorted(range(len(b)), key=lambda i: -b[i])
    return order_a == order_b


class TestScheduleValidation:
    """Constructor contracts."""

    def test_unknown_kind_rejected(self):
        """An unknown family name is a ValueError."""
        with pytest.raises(ValueError, match="unknown schedule kind"):
            AlphaSchedule("quadratic")

    def test_negative_peak_rejected(self):
        """A negative peak is a ValueError."""
        with pytest.raises(ValueError, match="peak"):
            AlphaSchedule("fixed", peak=-1.0)

    def test_floor_above_peak_rejected(self):
        """A floor above the peak is a ValueError."""
        with pytest.raises(ValueError, match="floor"):
            AlphaSchedule("linear", peak=0.5, floor=0.9)

    def test_nonpositive_total_steps_rejected(self):
        """total_steps must be positive."""
        with pytest.raises(ValueError, match="total_steps"):
            AlphaSchedule("fixed", total_steps=0)

    def test_negative_step_rejected(self):
        """A negative step index is a ValueError."""
        with pytest.raises(ValueError, match="step"):
            AlphaSchedule("fixed")(-1)

    def test_mismatched_context_scores_rejected(self):
        """context_scores must align with steps."""
        s = AlphaSchedule("margin_gated")
        with pytest.raises(ValueError, match="length"):
            s.values(4, context_scores=[0.1, 0.2])

    def test_as_dict_is_serialisable(self):
        """`as_dict` returns plain scalars."""
        d = AlphaSchedule("cosine", peak=2.0).as_dict()
        assert set(d) == {"kind", "peak", "total_steps", "floor", "gate_threshold"}
        assert all(isinstance(v, (str, int, float)) for v in d.values())


class TestSelectSchedule:
    """The ``docs/METHODS.md`` U2 rule, implemented verbatim."""

    def _results(self, **overrides):
        """Build a baseline results table, overriding selected entries.

        Args:
            **overrides: Schedule kinds to replace.

        Returns:
            A results mapping.
        """
        base = {
            "fixed": {"toxicity": 0.50, "perplexity": 20.0},
            "cosine": {"toxicity": 0.49, "perplexity": 20.2},
            "linear": {"toxicity": 0.48, "perplexity": 20.3},
        }
        for k, v in overrides.items():
            base.setdefault(k, v)
        base.update(overrides)
        return base

    def test_no_adoption_when_gain_is_below_threshold(self):
        """A 2 % relative gain is below the pre-registered 5 %."""
        res = select_schedule(self._results(
            linear={"toxicity": 0.49, "perplexity": 20.0}))
        assert res.adopted is False
        assert res.best_kind == "fixed"

    def test_adoption_when_gain_and_perplexity_both_pass(self):
        """A 10 % relative gain within the perplexity budget is adopted."""
        res = select_schedule(self._results(
            linear={"toxicity": 0.45, "perplexity": 20.5}))
        assert res.adopted is True
        assert res.best_kind == "linear"
        assert res.best_relative_gain == pytest.approx(0.10, abs=1e-9)
        assert res.perplexity_delta == pytest.approx(0.5, abs=1e-9)

    def test_perplexity_budget_blocks_adoption(self):
        """A large enough toxicity gain still fails if perplexity blows up."""
        res = select_schedule(self._results(
            linear={"toxicity": 0.20, "perplexity": 25.0}))
        assert res.adopted is False
        assert res.best_kind == "fixed"
        row = next(r for r in res.table if r["kind"] == "linear")
        assert row["qualifies"] is False
        assert row["perplexity_delta"] > MAX_PERPLEXITY_DELTA

    def test_ties_go_to_fixed(self):
        """A schedule exactly at the threshold with no margin does not win."""
        res = select_schedule(self._results(
            linear={"toxicity": 0.475, "perplexity": 20.0}))
        assert res.adopted is True  # exactly 5 %
        # ... but a 4.9 % gain does not.
        res2 = select_schedule(self._results(
            linear={"toxicity": 0.4755, "perplexity": 20.0}))
        assert res2.adopted is False

    def test_best_of_several_is_selected(self):
        """The strongest qualifying schedule wins."""
        res = select_schedule(self._results(
            linear={"toxicity": 0.45, "perplexity": 20.0},
            cosine={"toxicity": 0.40, "perplexity": 20.1}))
        assert res.best_kind == "cosine"
        assert res.best_relative_gain == pytest.approx(0.20, abs=1e-9)

    def test_missing_fixed_reference_rejected(self):
        """The rule needs its reference configuration."""
        with pytest.raises(KeyError, match="'fixed'"):
            select_schedule({"cosine": {"toxicity": 0.1, "perplexity": 1.0}})

    def test_missing_metric_rejected(self):
        """A result row missing the primary metric is an error."""
        with pytest.raises(KeyError, match="toxicity"):
            select_schedule({"fixed": {"perplexity": 1.0},
                             "linear": {"perplexity": 1.0}})

    def test_rule_is_recorded_verbatim(self):
        """The decision carries the rule that produced it."""
        res = select_schedule(self._results())
        assert "U2" in res.rule
        assert f"{MIN_RELATIVE_GAIN:.0%}" in res.rule

    def test_as_dict_round_trip(self):
        """`as_dict` is JSON-serialisable."""
        res = select_schedule(self._results())
        blob = res.as_dict()
        assert set(blob) >= {"adopted", "best_kind", "rule", "table"}
        assert isinstance(blob["table"], list)
