"""Attribute probes and confound diagnosis.

The confound
------------
A toxicity subspace fitted on hidden states is only a toxicity subspace if the
representation contains a toxicity direction to find. It might instead latch
onto something merely *correlated* with the labels in the labelled set. The two
that matter most, and that are both obvious in hindsight, are:

* **Refusal.** Models that decline a request produce distinctive text. A
  contrast set built from harmful prompts answered with refusals and from
  harmless prompts answered directly will be partly separable for reasons that
  have nothing to do with harm.
* **Curation style.** A contrast set assembled by hand is stylistically
  distinct from one assembled by sampling, so a probe can separate the two
  sets perfectly while separating nothing semantic. Appendix E flagged this as
  a live alternative explanation for the below-chance layer sweep.

What this module does
---------------------
It gives both confounds a name, a fitted direction, and a number:

* :func:`confound_angles` reports the principal angle between a target
  direction and each candidate confound. A near-perpendicular pair means the
  probe is not measuring the confound.
* :func:`partial_out` removes the component of a direction that lies along a
  confound, giving a confound-adjusted direction, and
  :func:`partial_out_heldout_accuracy` reports what that adjustment cost on
  held-out data.

The point is diagnostic, not corrective: the honest output is usually "this
direction is 12 degrees off the refusal axis, which is a lot", not a clean
number. An adjustment that is applied without reporting its cost is a way of
hiding a problem, so both numbers are always returned together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "Probe",
    "ConfoundReport",
    "confound_angles",
    "partial_out",
    "partial_out_heldout_accuracy",
    "PROBE_PRESETS",
]

#: Named probe directions that can be fitted on a labelled contrast set. Each
#: entry documents what the direction is *supposed* to mean, which is the point
#: at which a deviation becomes evidence of a confound.
PROBE_PRESETS: Dict[str, str] = {
    "toxicity": "harmful or abusive content; the intended target",
    "refusal": "declining or deflecting a request rather than complying",
    "formality": "register, politeness, and syntactic formality",
    "hedging": "epistemic uncertainty and non-committal phrasing",
    "first_person": "use of first-person pronouns; a pure lexical control",
}


@dataclass
class Probe:
    """A fitted linear direction over a model's representation space.

    Attributes:
        name: Probe name, ideally one of :data:`PROBE_PRESETS`.
        weight: Unit direction in feature space, shape ``(d,)``.
        bias: Threshold, shape ``(d,)`` or scalar; the sign convention is that
            positive scores are the positive class.
        heldout_accuracy: Balanced accuracy of the probe on held-out data.
        n_fit: Number of examples used to fit the probe.
        n_heldout_pos: Held-out positive-class examples.
        n_heldout_neg: Held-out negative-class examples.
        intent: The documented meaning of the probe, copied from
            :data:`PROBE_PRESETS` when the name is known.
    """

    name: str
    weight: torch.Tensor
    bias: torch.Tensor
    heldout_accuracy: float
    n_fit: int
    n_heldout_pos: int = 0
    n_heldout_neg: int = 0
    intent: str = ""

    @classmethod
    def fit(
        cls,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
        name: str,
        shrinkage: Optional[float] = None,
        solver: str = "cholesky",
        holdout_frac: float = 0.3,
        seed: int = 0,
    ) -> "Probe":
        """Fit a probe direction, holding out part of the data to score it.

        Args:
            positive_features: Embeddings of the positive class, shape ``(N1, d)``.
            negative_features: Embeddings of the negative class, shape ``(N2, d)``.
            name: Probe name. Recognised names get their intent recorded.
            shrinkage: Shrinkage coefficient for the shared covariance, or
                ``None`` for the scale-aware default.
            solver: Linear solver.
            holdout_frac: Fraction reserved for evaluation.
            seed: RNG seed for the split.

        Returns:
            A fitted :class:`Probe` with a unit-norm ``weight``.

        Raises:
            ValueError: If the two feature matrices disagree on width or either
                has fewer than two rows.
        """
        from .numerics import fit_direction, heldout_separability

        if positive_features.shape[1] != negative_features.shape[1]:
            raise ValueError(
                f"width mismatch: {positive_features.shape[1]} vs "
                f"{negative_features.shape[1]}"
            )
        if positive_features.shape[0] < 2 or negative_features.shape[0] < 2:
            raise ValueError("each class needs at least two examples")

        pos, neg = positive_features, negative_features
        g = torch.Generator().manual_seed(seed)
        n_fit_total = 0
        pos_idx = torch.randperm(pos.shape[0], generator=g)
        neg_idx = torch.randperm(neg.shape[0], generator=g)
        cut_p = max(1, min(pos.shape[0] - 1, int(round((1 - holdout_frac) * pos.shape[0]))))
        cut_n = max(1, min(neg.shape[0] - 1, int(round((1 - holdout_frac) * neg.shape[0]))))
        pos_tr, neg_tr = pos_idx[:cut_p], neg_idx[:cut_n]
        n_fit_total = int(pos_tr.numel() + neg_tr.numel())

        result = fit_direction(
            pos[pos_tr], neg[neg_tr], shrinkage=shrinkage, solver=solver
        )
        w = result.weight
        w = w / (w.norm() + 1e-12)
        acc = heldout_separability(
            result.weight, result.bias,
            pos[pos_idx[cut_p:]], neg[neg_idx[cut_n:]],
        )
        return cls(
            name=name,
            weight=w,
            bias=result.bias,
            heldout_accuracy=acc,
            n_fit=n_fit_total,
            n_heldout_pos=int(pos_idx[cut_p:].numel()),
            n_heldout_neg=int(neg_idx[cut_n:].numel()),
            intent=PROBE_PRESETS.get(name, "custom probe; intent undocumented"),
        )

    def z_vs_chance(self) -> float:
        """One-sided z-score of held-out accuracy against chance.

        Balanced accuracy under the null is a mean of two binomial proportions,
        so its variance is ``0.25 * (1/n_pos + 1/n_neg)``. A fixed accuracy
        threshold would be a magic number: on 240 held-out examples, a probe
        fitted to pure noise lands anywhere in roughly ``[0.38, 0.54]``, so any
        cutoff near 0.5 silently admits noise. Testing the hypothesis directly
        avoids that.

        Returns:
            The z-score; negative means worse than chance. ``0.0`` when either
            class has too few held-out examples for the test to be meaningful.
        """
        n_pos, n_neg = self.n_heldout_pos, self.n_heldout_neg
        if n_pos < 2 or n_neg < 2:
            return 0.0
        var = 0.25 * (1.0 / n_pos + 1.0 / n_neg)
        return (self.heldout_accuracy - 0.5) / math.sqrt(var)

    def is_identifiable(self, alpha: float = 0.05) -> bool:
        """Whether the probe is significantly better than chance.

        Args:
            alpha: One-sided significance level.

        Returns:
            ``True`` when the null of chance performance is rejected.
        """
        if self.n_heldout_pos < 2 or self.n_heldout_neg < 2:
            return False
        return self.z_vs_chance() > _normal_ppf(1.0 - alpha)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Score features with the probe.

        Args:
            features: Embeddings, shape ``(..., d)``.

        Returns:
            Tensor of shape ``(...)``.
        """
        flat = features.reshape(-1, self.weight.shape[0])
        out = flat @ self.weight
        return out.reshape(*features.shape[:-1])

    def as_dict(self) -> Dict[str, object]:
        """Return the probe metadata as a plain dictionary."""
        return {
            "name": self.name,
            "heldout_accuracy": self.heldout_accuracy,
            "z_vs_chance": self.z_vs_chance(),
            "identifiable": self.is_identifiable(),
            "n_fit": self.n_fit,
            "n_heldout_pos": self.n_heldout_pos,
            "n_heldout_neg": self.n_heldout_neg,
            "intent": self.intent,
        }


@dataclass
class ConfoundReport:
    """How far a target direction sits from each candidate confound.

    Attributes:
        target: The target probe name.
        confounds: One entry per candidate confound.
        verdict: ``"clean"``, ``"partial"`` or ``"confounded"``, assigned by
            comparing the smallest angle against
            :data:`CONFOUND_ANGLE_THRESHOLDS`.
    """

    target: str
    confounds: List[Dict[str, object]]
    verdict: str

    def worst(self) -> Optional[Dict[str, object]]:
        """Return the most-aligned confound, or None when there are none.

        Returns:
            The confound entry with the smallest principal angle.
        """
        if not self.confounds:
            return None
        return min(self.confounds, key=lambda c: c["angle_degrees"])

    def as_dict(self) -> Dict[str, object]:
        """Return the report as a plain dictionary."""
        return {"target": self.target, "verdict": self.verdict,
                "confounds": self.confounds}


def _normal_ppf(q: float) -> float:
    """Inverse standard-normal CDF, adequate for a significance cutoff.

    Uses the Acklam rational approximation; the error is below 1e-9 in the
    region of interest, which is far tighter than any threshold this module
    depends on.

    Args:
        q: Probability in ``(0, 1)``.

    Returns:
        The corresponding z-score.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        t = math.sqrt(-2 * math.log(q))
        return (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / \
               ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    if q > phigh:
        t = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / \
               ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    t = q - 0.5
    r = t * t
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * t / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


#: Angle thresholds for the verdict, in degrees. Below 15 deg the two
#: directions are near-parallel and the probe is measuring the confound; above
#: 45 deg they are near-orthogonal and the probe is measuring something else.
CONFOUND_ANGLE_THRESHOLDS = {"confounded": 15.0, "clean": 45.0}


def _unit(v: torch.Tensor) -> torch.Tensor:
    """Return ``v`` scaled to unit L2 norm.

    Args:
        v: A vector.

    Returns:
        The unit vector.
    """
    return v / (v.norm() + 1e-12)


def confound_angles(
    target: Probe,
    confounds: Sequence[Probe],
) -> ConfoundReport:
    """Measure the principal angle between a target and candidate confounds.

    Args:
        target: The probe of interest.
        confounds: Candidate confound probes, all in the same feature space.

    Returns:
        A :class:`ConfoundReport`. Probes whose own held-out accuracy is below
        0.5 (worse than chance) are reported but flagged, because the angle to
        a direction that carries no signal is not informative.

    Raises:
        ValueError: If a probe has a different feature width than the target.
    """
    d = target.weight.shape[0]
    rows: List[Dict[str, object]] = []
    for c in confounds:
        if c.weight.shape[0] != d:
            raise ValueError(
                f"probe {c.name!r} has width {c.weight.shape[0]}, "
                f"target has {d}"
            )
        cosine = float(_unit(target.weight) @ _unit(c.weight))
        clamped = max(-1.0, min(1.0, cosine))
        angle = math.degrees(math.acos(clamped))
        rows.append({
            "name": c.name,
            "angle_degrees": angle,
            "cosine": cosine,
            "confound_heldout_accuracy": c.heldout_accuracy,
            "confound_z_vs_chance": c.z_vs_chance(),
            "confound_is_usable": c.is_identifiable(),
            "intent": c.intent,
        })

    usable = [r for r in rows if r["confound_is_usable"]]
    if not usable:
        verdict = "clean"
    else:
        smallest = min(r["angle_degrees"] for r in usable)
        if smallest < CONFOUND_ANGLE_THRESHOLDS["confounded"]:
            verdict = "confounded"
        elif smallest > CONFOUND_ANGLE_THRESHOLDS["clean"]:
            verdict = "clean"
        else:
            verdict = "partial"
    rows.sort(key=lambda r: r["angle_degrees"])
    return ConfoundReport(target=target.name, confounds=rows, verdict=verdict)


def partial_out(
    target: Probe,
    confound: Probe,
    renormalise: bool = True,
) -> torch.Tensor:
    """Remove the confound's component from a direction.

    This is Gram--Schmidt against the confound: the returned direction is
    orthogonal to the confound by construction.

    Args:
        target: The direction to adjust.
        confound: The direction to remove.
        renormalise: Return a unit vector.

    Returns:
        The adjusted direction, shape ``(d,)``.

    Raises:
        ValueError: If the widths differ, or the two directions are parallel to
            within numerical precision -- in which case the adjusted direction
            is zero and there is nothing to report but a number.
    """
    if target.weight.shape != confound.weight.shape:
        raise ValueError(
            f"width mismatch: {target.weight.shape[0]} vs "
            f"{confound.weight.shape[0]}"
        )
    u = _unit(confound.weight)
    proj = (target.weight @ u) * u
    adjusted = target.weight - proj
    residual = float(adjusted.norm())
    if residual < 1e-8:
        raise ValueError(
            f"target {target.name!r} is parallel to confound "
            f"{confound.name!r}; nothing survives partial removal"
        )
    return _unit(adjusted) if renormalise else adjusted


def partial_out_heldout_accuracy(
    target: Probe,
    confound: Probe,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Report what removing a confound costs, on held-out data.

    The cost is reported alongside the adjusted accuracy deliberately: an
    adjustment that is applied without its price is a way of concealing that
    the original direction was leaning on the confound.

    Args:
        target: The probe being adjusted.
        confound: The probe being removed.
        positive_features: Held-out positive-class embeddings.
        negative_features: Held-out negative-class embeddings.
        bias: Bias to use. Defaults to the target's own.

    Returns:
        A dict with ``accuracy_before``, ``accuracy_after``, ``cost`` and
        ``cosine_removed``.
    """
    from .numerics import heldout_separability

    b = target.bias if bias is None else bias
    before = heldout_separability(
        target.weight, b, positive_features, negative_features
    )
    adjusted = partial_out(target, confound, renormalise=True)
    after = heldout_separability(
        adjusted, b, positive_features, negative_features
    )
    cosine_removed = abs(float(_unit(target.weight) @ _unit(confound.weight)))
    return {
        "accuracy_before": before,
        "accuracy_after": after,
        "cost": before - after,
        "cosine_removed": cosine_removed,
    }
