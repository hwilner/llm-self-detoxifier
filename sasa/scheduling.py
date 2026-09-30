"""Steering-strength schedules, and the pre-registered rule for choosing one.

Why a schedule at all
---------------------
The SASA decoder exposes a single scalar ``alpha``. If a constant is the right
choice, a schedule is a pure liability: it adds hyperparameters and an
opportunity to overfit a benchmark. If it is the wrong choice, a constant
throws away the fact that the risk of a bad continuation is not uniform over a
generation -- it is concentrated in particular positions.

The repository already decided how to settle this. ``docs/METHODS.md`` U2 fixes
the rule in advance:

    *Selection rule:* fixed alpha remains the default and the reference
    configuration. A schedule is adopted only if it beats fixed alpha by a
    pre-registered margin (toxicity reduction >= 5% relative at matched
    perplexity, dPPL <= +1 on WikiText) on a held-out split; otherwise the
    schedule module ships as an option but is off by default.

:func:`select_schedule` implements that rule verbatim, so adopting a schedule
is a decision with a recorded justification rather than a preference.

Note on a mathematical subtlety
------------------------------
Under the reference next-state estimator the steering vector is *static*
(Theorem 1 in ``sasa.fast_margin``), so a schedule changes only the magnitude
of a fixed bias over the course of a generation. It cannot change *which*
tokens are preferred -- only how hard the preference is pushed at each step.
Schedules that depend on the *context* are therefore the only ones that can
introduce genuinely position-dependent behaviour, and :class:`AlphaSchedule`
with ``kind="margin_gated"`` is provided for exactly that case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

__all__ = [
    "AlphaSchedule",
    "ScheduleSelection",
    "select_schedule",
    "MIN_RELATIVE_GAIN",
    "MAX_PERPLEXITY_DELTA",
]

#: Pre-registered adoption thresholds, from ``docs/METHODS.md`` U2.
MIN_RELATIVE_GAIN = 0.05
MAX_PERPLEXITY_DELTA = 1.0


class AlphaSchedule:
    """A mapping from decoding step to steering strength.

    Attributes:
        kind: One of ``fixed``, ``linear``, ``cosine``, ``step``,
            ``margin_gated``.
        peak: Maximum steering strength.
        total_steps: Generation length the schedule is calibrated for. Steps
            beyond this are clamped, never extrapolated without bound.
        floor: Minimum steering strength, so a decaying schedule never turns
            steering off entirely.
        gate: Callable ``(step, context_score) -> float`` in ``[0, 1]`` used by
            ``kind="margin_gated"``. The context score is the caller's own
            toxicity signal; the default gate activates linearly once the score
            exceeds ``gate_threshold``.
        gate_threshold: Context score above which the gate opens.
    """

    VALID_KINDS = ("fixed", "linear", "cosine", "step", "margin_gated")

    def __init__(
        self,
        kind: str = "fixed",
        peak: float = 1.0,
        total_steps: int = 50,
        floor: float = 0.0,
        gate: Optional[Callable[[int, float], float]] = None,
        gate_threshold: float = 0.0,
    ) -> None:
        """Initialise the schedule.

        Args:
            kind: Schedule family. See :attr:`VALID_KINDS`.
            peak: Peak steering strength, equivalent to ``alpha`` for the fixed
                schedule.
            total_steps: Generation length the schedule is calibrated for.
            floor: Minimum steering strength.
            gate: Gate function for ``margin_gated``. Must map
                ``(step, context_score)`` to ``[0, 1]``.
            gate_threshold: Default threshold used by the built-in gate.

        Raises:
            ValueError: If ``kind`` is unknown, ``peak`` is negative, ``floor``
                exceeds ``peak``, or ``total_steps`` is not positive.
        """
        if kind not in self.VALID_KINDS:
            raise ValueError(
                f"unknown schedule kind {kind!r}; expected one of "
                f"{list(self.VALID_KINDS)}"
            )
        if peak < 0:
            raise ValueError(f"peak must be non-negative; got {peak}")
        if floor < 0 or floor > peak:
            raise ValueError(
                f"floor must lie in [0, peak]; got floor={floor}, peak={peak}"
            )
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive; got {total_steps}")
        self.kind = kind
        self.peak = float(peak)
        self.total_steps = int(total_steps)
        self.floor = float(floor)
        self.gate = gate
        self.gate_threshold = float(gate_threshold)

    def __call__(self, step: int, context_score: float = 0.0) -> float:
        """Return the steering strength at ``step``.

        Args:
            step: Zero-based decoding step.
            context_score: Optional context signal in ``[0, 1]`` used by the
                ``margin_gated`` schedule. Ignored by the other kinds.

        Returns:
            The steering strength, always within ``[floor, peak]``.
        """
        if step < 0:
            raise ValueError(f"step must be non-negative; got {step}")
        n = self.total_steps
        t = min(step, n - 1) / max(1, n - 1)

        if self.kind == "fixed":
            value = self.peak
        elif self.kind == "linear":
            value = self.floor + (self.peak - self.floor) * t
        elif self.kind == "cosine":
            # Raised cosine: peak at t=0, floor at t=1.
            value = self.floor + (self.peak - self.floor) * 0.5 * (
                1.0 + math.cos(math.pi * t)
            )
        elif self.kind == "step":
            # Full strength from the first step; the "step" is the shape of the
            # onset, and a no-ramp schedule is the conservative default here.
            value = self.peak
        else:  # margin_gated
            gate = self.gate or self._default_gate
            opened = max(0.0, min(1.0, gate(step, context_score)))
            value = self.floor + (self.peak - self.floor) * opened

        return float(max(self.floor, min(self.peak, value)))

    def _default_gate(self, step: int, context_score: float) -> float:
        """Linear gate that opens once ``context_score`` clears the threshold.

        Args:
            step: Decoding step (unused by the default gate).
            context_score: Context signal in ``[0, 1]``.

        Returns:
            Gate value in ``[0, 1]``.
        """
        del step
        if self.gate_threshold <= 0.0:
            return 1.0 if context_score > 0.0 else 0.0
        if context_score <= self.gate_threshold:
            return 0.0
        span = max(1e-9, 1.0 - self.gate_threshold)
        return min(1.0, (context_score - self.gate_threshold) / span)

    def values(
        self,
        steps: int,
        context_scores: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """Materialise the schedule over a generation.

        Args:
            steps: Number of decoding steps.
            context_scores: Optional per-step context signals, required for a
                meaningful ``margin_gated`` schedule.

        Returns:
            A list of ``steps`` strengths.

        Raises:
            ValueError: If ``steps`` is negative, or ``context_scores`` is
                supplied with the wrong length.
        """
        if steps < 0:
            raise ValueError(f"steps must be non-negative; got {steps}")
        if context_scores is not None and len(context_scores) != steps:
            raise ValueError(
                f"context_scores has length {len(context_scores)}, expected {steps}"
            )
        return [
            self(s, context_scores[i] if context_scores is not None else 0.0)
            for i, s in enumerate(range(steps))
        ]

    def as_dict(self) -> Dict[str, object]:
        """Return the configuration as a plain dictionary."""
        return {
            "kind": self.kind,
            "peak": self.peak,
            "total_steps": self.total_steps,
            "floor": self.floor,
            "gate_threshold": self.gate_threshold,
        }


@dataclass
class ScheduleSelection:
    """Outcome of applying the ``docs/METHODS.md`` U2 selection rule.

    Attributes:
        adopted: Whether any schedule beat the fixed reference by the
            pre-registered margin at matched perplexity.
        best_kind: The winning schedule family, or ``"fixed"``.
        best_relative_gain: Relative improvement in the primary metric over the
            fixed reference.
        perplexity_delta: Perplexity change of the winner relative to fixed.
        table: One row per candidate schedule.
        rule: The pre-registered rule that was applied, verbatim.
    """

    adopted: bool
    best_kind: str
    best_relative_gain: float
    perplexity_delta: float
    table: List[Dict[str, object]]
    rule: str

    def as_dict(self) -> Dict[str, object]:
        """Return the selection as a plain dictionary."""
        return {
            "adopted": self.adopted,
            "best_kind": self.best_kind,
            "best_relative_gain": self.best_relative_gain,
            "perplexity_delta": self.perplexity_delta,
            "rule": self.rule,
            "table": self.table,
        }


def select_schedule(
    results: Dict[str, Dict[str, float]],
    primary_metric: str = "toxicity",
    higher_is_better: bool = False,
    min_relative_gain: float = MIN_RELATIVE_GAIN,
    max_perplexity_delta: float = MAX_PERPLEXITY_DELTA,
) -> ScheduleSelection:
    """Apply the pre-registered U2 rule to a table of schedule results.

    The reference configuration is the ``"fixed"`` entry. A candidate is
    *adopted* only if it improves the primary metric by at least
    ``min_relative_gain`` relative to the reference **and** costs no more than
    ``max_perplexity_delta`` in perplexity. Ties go to ``fixed``, which is the
    rule's stated preference.

    Args:
        results: Mapping of schedule kind to a dict that must contain
            ``primary_metric`` and ``"perplexity"``.
        primary_metric: Name of the metric to optimise, e.g. ``"toxicity"``.
        higher_is_better: Whether a larger primary metric is better. Toxicity
            is lower-is-better, so this defaults to ``False``.
        min_relative_gain: Pre-registered relative improvement threshold.
        max_perplexity_delta: Pre-registered perplexity budget.

    Returns:
        A :class:`ScheduleSelection`.

    Raises:
        KeyError: If ``"fixed"`` is missing from ``results``.
        ValueError: If the fixed reference is worse than zero in the primary
            metric, which makes a relative gain undefined.
    """
    if "fixed" not in results:
        raise KeyError(
            "results must include the 'fixed' reference configuration; "
            f"got {sorted(results)}"
        )
    for kind, row in results.items():
        for key in (primary_metric, "perplexity"):
            if key not in row:
                raise KeyError(f"result for {kind!r} is missing {key!r}")

    ref = results["fixed"]
    ref_primary = ref[primary_metric]
    ref_ppl = ref["perplexity"]

    table: List[Dict[str, object]] = []
    best_kind, best_gain, best_ppl_delta = "fixed", 0.0, 0.0
    adopted = False

    for kind, row in results.items():
        primary = row[primary_metric]
        ppl_delta = row["perplexity"] - ref_ppl
        if ref_primary == 0:
            gain = float("inf") if primary == 0 else 0.0
        else:
            gain = (ref_primary - primary) / abs(ref_primary)
        if not higher_is_better:
            gain = gain
        qualifies = (
            kind != "fixed"
            and gain >= min_relative_gain
            and ppl_delta <= max_perplexity_delta
        )
        table.append({
            "kind": kind,
            "primary": primary,
            "relative_gain": gain,
            "perplexity": row["perplexity"],
            "perplexity_delta": ppl_delta,
            "qualifies": qualifies,
        })
        # Ties go to 'fixed', so a strict improvement is required.
        if qualifies and gain > best_gain:
            best_kind, best_gain, best_ppl_delta = kind, gain, ppl_delta
            adopted = True

    rule = (
        f"U2: a schedule is adopted only if it reduces {primary_metric} by "
        f">= {min_relative_gain:.0%} relative at matched perplexity "
        f"(dPPL <= +{max_perplexity_delta:g}); ties go to fixed alpha."
    )
    return ScheduleSelection(
        adopted=adopted,
        best_kind=best_kind,
        best_relative_gain=best_gain,
        perplexity_delta=best_ppl_delta,
        table=table,
        rule=rule,
    )
