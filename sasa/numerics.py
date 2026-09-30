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
        weight: The decision direction ``w``, shape ``(d,)``. Always returned in
            the *original* feature space, even when a PCA pre-conditioner was
            used, so downstream code needs no special case.
        bias: The decision bias ``b``, shape ``(d,)``.
        mu_neg: Class mean of the non-toxic (positive) class, shape ``(d,)``.
        mu_pos: Class mean of the toxic (negative) class, shape ``(d,)``.
        sigma: The regularised shared covariance actually used, shape ``(d, d)``
            in the reduced space when ``basis`` is not None, else ``(d, d)``.
        solver: Name of the solver that produced ``weight``.
        shrinkage: The shrinkage coefficient that was applied.
        raw_condition: Condition number of the unregularised covariance.
        reg_condition: Condition number of the regularised covariance.
        basis: Optional PCA projection ``(k, d)`` used before the solve, or
            None. Applying it to a feature matrix performs the reduction.
        effective_dim: Dimensionality of the space the solve happened in.
        explained_variance: Fraction of pooled variance retained by ``basis``.
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
    basis: Optional[torch.Tensor] = None
    effective_dim: Optional[int] = None
    explained_variance: Optional[float] = None

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Map features into the reduced space the solve happened in.

        Args:
            features: Feature matrix, shape ``(N, d)``.

        Returns:
            The reduced features, shape ``(N, k)``; unchanged when no
            pre-conditioner was used.
        """
        if self.basis is None:
            return features
        return features @ self.basis.T

    def as_dict(self) -> Dict[str, float]:
        """Return scalar diagnostics as a plain dictionary."""
        return {
            "solver": self.solver,
            "shrinkage": self.shrinkage,
            "raw_condition": self.raw_condition,
            "reg_condition": self.reg_condition,
            "effective_dim": self.effective_dim or self.weight.shape[0],
            "explained_variance": self.explained_variance,
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
    pca_dim: Optional[int] = None,
) -> FitResult:
    """Fit the SASA decision direction with a switchable solver and regulariser.

    ``shrinkage=None`` selects the data-driven Ledoit-Wolf coefficient, scaled
    by ``trace(Sigma)/d`` so the penalty is invariant to the overall scale of the
    features. Passing a float uses that value instead; passing ``0.0``
    reproduces an unregularised fit.

    ``pca_dim`` is the important one for real hidden states. When the labelled
    set is much smaller than the hidden size -- which is the normal case for
    LLM activations, where ``d`` is 768 to 4096 and a realistic labelled corpus
    is tens to hundreds of examples -- the pooled covariance has rank at most
    ``N1 + N2 - 2`` and is therefore singular. Solving against it yields a
    direction that interpolates the training set exactly and can generalise
    worse than chance. Passing ``pca_dim <= min(N1+N2-2, d)`` projects onto the
    leading principal directions first, which restores a well-posed problem.
    The returned ``weight`` is mapped back to the original feature space.

    Args:
        embeddings_neg: Non-toxic embeddings, shape ``(N1, d)``.
        embeddings_pos: Toxic embeddings, shape ``(N2, d)``.
        shrinkage: Shrinkage coefficient in ``[0, 1]``, or ``None`` for
            Ledoit-Wolf estimation.
        solver: Solver passed to :func:`solve_direction`.
        ridge: Absolute ridge used only when no shrinkage is applied; keeps the
            matrix positive definite.
        pca_dim: Number of principal directions to keep, or ``None`` to solve
            in the full space.

    Returns:
        A :class:`FitResult`.

    Raises:
        ValueError: If inputs are shape-incompatible, ``solver`` is unknown, or
            ``pca_dim`` exceeds the number of available samples.
    """
    sigma_raw_full = shared_covariance(embeddings_neg, embeddings_pos)
    d = sigma_raw_full.shape[0]
    n = embeddings_neg.shape[0] + embeddings_pos.shape[0]
    mu_neg = embeddings_neg.mean(0)
    mu_pos = embeddings_pos.mean(0)

    basis: Optional[torch.Tensor] = None
    explained: Optional[float] = None
    if pca_dim is not None:
        max_dim = min(n - 2, d)
        if pca_dim > max_dim:
            raise ValueError(
                f"pca_dim={pca_dim} exceeds the rank available from {n} "
                f"samples (max {max_dim}); the fit would be singular"
            )
        # Directions come from the *within-class centred* pool, so the basis
        # spans the discriminative variation rather than the global mean shift.
        pooled = torch.cat(
            [embeddings_neg - mu_neg, embeddings_pos - mu_pos], dim=0
        )
        _, svals, vh = torch.linalg.svd(pooled, full_matrices=False)
        basis = vh[:pca_dim].contiguous()
        total = float((svals ** 2).sum())
        explained = float((svals[:pca_dim] ** 2).sum() / total) if total > 0 else None

    if basis is None:
        # Project the raw features; the LDA step re-centres internally.
        neg = embeddings_neg
        pos = embeddings_pos
    else:
        neg = embeddings_neg @ basis.T
        pos = embeddings_pos @ basis.T

    sigma_raw = (neg - neg.mean(0)).T @ (neg - neg.mean(0))
    sigma_raw = sigma_raw + (pos - pos.mean(0)).T @ (pos - pos.mean(0))
    sigma_raw = sigma_raw / (n - 2)

    k = sigma_raw.shape[0]
    lam = (
        float(shrinkage) if shrinkage is not None
        else ledoit_wolf_shrinkage(sigma_raw, n)
    )
    target = torch.diagonal(sigma_raw).mean()
    sigma = sigma_raw + lam * target * torch.eye(k, dtype=sigma_raw.dtype,
                                                device=sigma_raw.device)
    if lam == 0.0:
        sigma = sigma + ridge * torch.eye(k, dtype=sigma_raw.dtype,
                                          device=sigma_raw.device)

    try:
        w_k = solve_direction(sigma, neg.mean(0) - pos.mean(0), solver=solver)
    except RuntimeError:
        if solver != "cholesky":
            raise
        sigma = sigma + ridge * torch.eye(k, dtype=sigma_raw.dtype,
                                          device=sigma_raw.device)
        w_k = solve_direction(sigma, neg.mean(0) - pos.mean(0), solver="cholesky")

    w = w_k if basis is None else (basis.T @ w_k)

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
        basis=basis,
        effective_dim=k,
        explained_variance=explained,
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
