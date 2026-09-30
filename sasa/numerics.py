"""Numerically robust fitting of the SASA decision direction.

The reference :meth:`sasa.subspace_learner.SubspaceLearner.fit` forms the
shared covariance as a sum of outer products, adds a fixed ``1e-6 * I`` ridge,
and calls :func:`torch.linalg.inv`. For the hidden sizes this project targets
that has two problems worth measuring rather than asserting:

* **Stability.** An explicit inverse squares the condition number. The estimator
  also uses a fixed ridge that is unrelated to the scale of the data, so on
  high-dimensional, strongly anisotropic hidden states the ridge is either
  negligible (leaving the solve ill-conditioned) or overwhelming (biasing the
  direction toward the identity).
* **Conditioning of the solve.** A Cholesky factorisation solves
  ``Sigma w = (mu_1 - mu_2)`` without ever forming ``Sigma^-1``, and the
  natural regulariser is a *scaled* ridge ``lambda * trace(Sigma) / d`` rather
  than an absolute constant.

This module provides the same closed-form estimator with three switchable
solvers and a held-out diagnostic, so the choice is made on evidence. It is
additive: :class:`sasa.subspace_learner.SubspaceLearner` is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

__all__ = [
    "FitResult",
    "shared_covariance",
    "fit_direction",
    "ledoit_wolf_shrinkage",
    "solve_direction",
    "heldout_separability",
]


@dataclass
class FitResult:
    """Output of a single subspace fit.

    Attributes:
        weight: The decision direction ``w``, shape ``(d,)``.
        bias: The decision bias ``b``, shape ``(d,)``.
        mu_neg: Class mean of the non-toxic (positive) class, shape ``(d,)``.
        mu_pos: Class mean of the toxic (negative) class, shape ``(d,)``.
        sigma: The regularised shared covariance actually used, shape
            ``(d, d)``.
        solver: Name of the solver that produced ``weight``.
        shrinkage: The shrinkage coefficient that was applied.
        raw_condition: Condition number of the unregularised covariance.
        reg_condition: Condition number of the regularised covariance.
    """

    weight: torch.Tensor
    bias: torch.Tensor
    mu_neg: torch.Tensor
    mu_pos: torch.Tensor
    sigma: torch.Tensor
    solver: str
    shrinkage: float
    raw_condition: float
    reg_condition: float

    def as_dict(self) -> Dict[str, float]:
        """Return scalar diagnostics as a plain dictionary."""
        return {
            "solver": self.solver,
            "shrinkage": self.shrinkage,
            "raw_condition": self.raw_condition,
            "reg_condition": self.reg_condition,
        }


def shared_covariance(
    embeddings_neg: torch.Tensor,
    embeddings_pos: torch.Tensor,
) -> torch.Tensor:
    """Pooled within-class covariance with the standard ``(N1+N2-2)`` scaling.

    Args:
        embeddings_neg: Non-toxic embeddings, shape ``(N1, d)``.
        embeddings_pos: Toxic embeddings, shape ``(N2, d)``.

    Returns:
        Unregularised pooled covariance, shape ``(d, d)``.

    Raises:
        ValueError: If either input has fewer than two rows or the two inputs
            disagree on the feature dimension.
    """
    if embeddings_neg.shape[1] != embeddings_pos.shape[1]:
        raise ValueError(
            f"feature mismatch: {embeddings_neg.shape[1]} vs "
            f"{embeddings_pos.shape[1]}"
        )
    if embeddings_neg.shape[0] < 2 or embeddings_pos.shape[0] < 2:
        raise ValueError("need at least 2 samples per class to estimate a covariance")

    mu_neg = embeddings_neg.mean(0)
    mu_pos = embeddings_pos.mean(0)
    c1 = embeddings_neg - mu_neg
    c2 = embeddings_pos - mu_pos
    dof = embeddings_neg.shape[0] + embeddings_pos.shape[0] - 2
    return (c1.T @ c1 + c2.T @ c2) / dof


def ledoit_wolf_shrinkage(sigma: torch.Tensor, n_samples: int) -> float:
    """Estimate a Ledoit-Wolf-style shrinkage coefficient toward a scaled identity.

    The coefficient is the ratio of the variance of the off-diagonal sample
    covariances to the squared average of the true covariance entries, clipped
    to ``[0, 1]``. It is the standard closed-form shrinkage for a covariance
    matrix in the linear-discriminant setting.

    Args:
        sigma: Unregularised pooled covariance, shape ``(d, d)``.
        n_samples: Effective number of samples, used for the finite-sample
            correction.

    Returns:
        A shrinkage coefficient in ``[0, 1]``.
    """
    d = sigma.shape[0]
    mu = torch.diagonal(sigma).mean()
    mu2 = (sigma ** 2).sum() / (d * d)
    denom = n_samples * (mu2 - mu * mu)
    if denom.abs() < 1e-12:
        return 1.0
    delta = (mu2 - (sigma ** 2).sum() / d) / denom
    return float(delta.clamp(0.0, 1.0).item())


def solve_direction(
    sigma: torch.Tensor,
    delta: torch.Tensor,
    solver: str = "cholesky",
) -> torch.Tensor:
    """Solve ``Sigma w = delta`` without forming the inverse where possible.

    Args:
        sigma: Regularised covariance, shape ``(d, d)``.
        delta: Right-hand side, shape ``(d,)``.
        solver: One of ``"cholesky"``, ``"solve"`` (LU/LAPACK ``solve``), or
            ``"inv"`` (explicit inverse, the reference behaviour, kept for
            comparison).

    Returns:
        The solution ``w``, shape ``(d,)``.

    Raises:
        ValueError: If ``solver`` is not one of the accepted values.
        RuntimeError: If a Cholesky factorisation fails; callers should fall
            back to ``"solve"`` or add shrinkage.
    """
    if solver == "cholesky":
        chol = torch.linalg.cholesky(sigma)
        return torch.cholesky_solve(delta.unsqueeze(-1), chol).squeeze(-1)
    if solver == "solve":
        return torch.linalg.solve(sigma, delta)
    if solver == "inv":
        return torch.linalg.inv(sigma) @ delta
    raise ValueError(
        f"unknown solver {solver!r}; expected 'cholesky', 'solve' or 'inv'"
    )


def fit_direction(
    embeddings_neg: torch.Tensor,
    embeddings_pos: torch.Tensor,
    shrinkage: Optional[float] = None,
    solver: str = "cholesky",
    ridge: float = 1e-6,
) -> FitResult:
    """Fit the SASA decision direction with a switchable solver and regulariser.

    ``shrinkage=None`` selects the data-driven Ledoit-Wolf coefficient, scaled
    by ``trace(Sigma)/d`` so the penalty is invariant to the overall scale of the
    features. Passing a float uses that value instead; passing ``0.0``
    reproduces an unregularised fit.

    Args:
        embeddings_neg: Non-toxic embeddings, shape ``(N1, d)``.
        embeddings_pos: Toxic embeddings, shape ``(N2, d)``.
        shrinkage: Shrinkage coefficient in ``[0, 1]``, or ``None`` for
            Ledoit-Wolf estimation.
        solver: Solver passed to :func:`solve_direction`.
        ridge: Absolute ridge used only when ``shrinkage is None`` *and*
            Ledoit-Wolf returns zero; keeps the matrix positive definite.

    Returns:
        A :class:`FitResult`.

    Raises:
        ValueError: If inputs are shape-incompatible or ``solver`` is unknown.
        RuntimeError: If the Cholesky factorisation fails even after
            regularisation.
    """
    sigma_raw = shared_covariance(embeddings_neg, embeddings_pos)
    d = sigma_raw.shape[0]
    n = embeddings_neg.shape[0] + embeddings_pos.shape[0]

    mu_neg = embeddings_neg.mean(0)
    mu_pos = embeddings_pos.mean(0)

    lam = (
        float(shrinkage) if shrinkage is not None
        else ledoit_wolf_shrinkage(sigma_raw, n)
    )
    target = torch.diagonal(sigma_raw).mean()
    sigma = sigma_raw + lam * target * torch.eye(d, dtype=sigma_raw.dtype,
                                                device=sigma_raw.device)
    if lam == 0.0:
        sigma = sigma + ridge * torch.eye(d, dtype=sigma_raw.dtype,
                                          device=sigma_raw.device)

    try:
        w = solve_direction(sigma, mu_neg - mu_pos, solver=solver)
    except RuntimeError:
        if solver != "cholesky":
            raise
        # Fall back to a slightly stronger ridge, then retry once.
        sigma = sigma + ridge * torch.eye(d, dtype=sigma_raw.dtype,
                                          device=sigma_raw.device)
        w = solve_direction(sigma, mu_neg - mu_pos, solver="cholesky")

    raw_cond = float(torch.linalg.cond(sigma_raw).item())
    reg_cond = float(torch.linalg.cond(sigma).item())

    return FitResult(
        weight=w,
        bias=0.5 * (mu_neg + mu_pos),
        mu_neg=mu_neg,
        mu_pos=mu_pos,
        sigma=sigma,
        solver=solver,
        shrinkage=lam,
        raw_condition=raw_cond,
        reg_condition=reg_cond,
    )


def heldout_separability(
    weight: torch.Tensor,
    bias: torch.Tensor,
    embeddings_neg: torch.Tensor,
    embeddings_pos: torch.Tensor,
) -> float:
    """Balanced accuracy of a fitted direction on held-out data.

    Args:
        weight: Decision direction, shape ``(d,)``.
        bias: Decision bias, shape ``(d,)``.
        embeddings_neg: Held-out non-toxic embeddings, shape ``(N1, d)``.
        embeddings_pos: Held-out toxic embeddings, shape ``(N2, d)``.

    Returns:
        Balanced accuracy in ``[0, 1]``, i.e. the mean of the two class
        recalls so that class imbalance cannot inflate the score.
    """
    s_neg = (embeddings_neg - bias) @ weight
    s_pos = (embeddings_pos - bias) @ weight
    recall_neg = (s_neg > 0).float().mean().item()
    recall_pos = (s_pos < 0).float().mean().item()
    return 0.5 * (recall_neg + recall_pos)
