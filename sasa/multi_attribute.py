r"""Graded multi-attribute subspaces, and the static geometry of their interference.

Two ideas meet in this module.

Rank-$k$ margins from graded labels (idea 8)
--------------------------------------------
SASA's margin is a *rank-one* construction: a single hyperplane
$w^\\top (h - b)$ separating toxic from non-toxic. Two-class Fisher
discriminant analysis can only ever produce one direction, so a rank-one margin
is not a simplification of a richer method -- it is all the two-class method
has. But the external judge (Appendix D) already emits a **graded** label on the
0-4 scale at no extra labelling cost. With $k$ classes, multiclass LDA has
$k-1$ independent discriminant directions, and the natural generalisation of
the SASA margin is

.. math::
    f(x) = \\max_{j \\in \\text{benign}} \\delta_j(x) \\;-\\;
           \\min_{j \\in \\text{toxic}} \\delta_j(x),
    \\qquad
    \\delta_j(x) = x^\\top \\Sigma^{-1}\\mu_j - \\tfrac12 \\mu_j^\\top \\Sigma^{-1}\\mu_j ,

which for $k = 2$ reduces *exactly* to the binary score
$\\delta_{\\text{benign}} - \\delta_{\\text{toxic}}$. Graded supervision is
therefore free, and it strictly contains the binary method.

Static interference (idea 6)
----------------------------
If attribute $i$ steers with a margin $f_i$ and attribute $j$ with $f_j$, the
composed decoder is $z' = z + \\sum_k \\alpha_k f_k$, and the *only* channel
through which they can interact is the overlap of the two margins' weight
subspaces. That overlap is computable once, from the fitted subspaces, with no
generation at all. The claim tested in Appendix H is that this **static
prediction ranks measured interference correctly** -- which, if true, turns a
combinatorial search over attribute schedules into a screening pass over
pairwise cosines.

Subspace overlap is measured by the mean squared principal cosine
(``subspace_overlap``), which reduces to the squared cosine for rank-one
directions, plus the smallest principal angle, which is the quantity that
actually determines whether the two margins can be simultaneously large.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .numerics import heldout_separability, shared_covariance, solve_direction

__all__ = [
    "GradedSubspace",
    "AttributeSet",
    "InterferencePair",
    "IDENTIFIABILITY_GATE",
]

#: Held-out balanced accuracy an attribute must clear to enter the interference
#: study. Pre-registered: an attribute whose own subspace is not identifiable
#: cannot have a meaningful interaction with another one, and including it would
#: measure noise (this is the failure demonstrated in Appendix E).
IDENTIFIABILITY_GATE = 0.60


@dataclass
class GradedSubspace:
    """A multiclass discriminant subspace fitted to graded labels.

    Attributes:
        name: Attribute name, e.g. ``"toxicity"``.
        mu_whitened: Class means in whitened coordinates, shape ``(k, d)``.
        delta_proj: Per-class offset ``-0.5 * mu_j^T Sigma^{-1} mu_j``, shape
            ``(k,)``.
        basis: The ``(r, d)`` weight subspace, where ``r = k - 1``.
        thresholds: Grade value of each class, ascending, shape ``(k,)``.
        toxic_from: Grades at or above this value define the toxic side.
        heldout_accuracy: Fraction of held-out examples whose predicted grade
            matches, under a nearest-class-mean rule in whitened space.
        heldout_side_accuracy: Fraction of held-out examples on the correct
            side of the toxic/non-toxic split. This is the number the
            identifiability gate is applied to.
        shrinkage: Shrinkage coefficient that was applied.
        whitener: The whitening operator ``Sigma^{-1}``, shape ``(d, d)``.
        embedding_dim: Feature width ``d``.
        n_fit: Number of examples used to fit.
    """

    name: str
    mu_whitened: torch.Tensor
    delta_proj: torch.Tensor
    basis: torch.Tensor
    thresholds: torch.Tensor
    toxic_from: float
    heldout_accuracy: float
    heldout_side_accuracy: float
    shrinkage: float
    whitener: torch.Tensor
    embedding_dim: int
    n_fit: int = 0
    spectrum: Sequence[float] = ()
    merged_grades: Tuple[float, ...] = ()

    # ------------------------------------------------------------------ fit

    @classmethod
    def fit(
        cls,
        features: torch.Tensor,
        grades: Sequence[float],
        name: str = "attribute",
        toxic_from: float = 2.0,
        shrinkage: Optional[float] = None,
        solver: str = "cholesky",
        holdout_frac: float = 0.3,
        seed: int = 0,
        rank_tol_rel: float = 0.1,
        min_class_size: int = 2,
    ) -> "GradedSubspace":
        """Fit a graded discriminant subspace.

        The whitening operator and the class means are estimated on the training
        split only; the split is a seeded permutation so the result is
        reproducible.

        Args:
            features: Labelled embeddings, shape ``(N, d)``.
            grades: Grade for each example, e.g. on ``[0, 4]``, length ``N``.
            name: Attribute name.
            toxic_from: Grades at or above this are on the toxic side.
            shrinkage: Shrinkage coefficient, or ``None`` to apply a small
                scale-aware default. Passing ``0.0`` applies no ridge.
            solver: Linear solver used for the whitening step.
            holdout_frac: Fraction reserved for evaluation.
            seed: RNG seed for the split.
            rank_tol_rel: Keep a discriminant direction only if its singular
                value is at least this fraction of the largest. Defaults to
                ``0.1``; see the rank-selection note in the implementation.
            min_class_size: Grades with fewer examples than this are merged
                into the nearest populated grade. Ordinal labels are naturally
                lumpy -- grades 1 and 3 often have a single example -- and a
                class with one example contributes no within-class scatter at
                all, so it cannot be fitted. The merge is recorded in
                ``merged_grades`` rather than applied silently.

        Returns:
            A fitted :class:`GradedSubspace`.

        Raises:
            ValueError: If the inputs are inconsistent, fewer than two distinct
                grades are present, or merging would leave a single class.
        """
        x = features.detach()
        g = torch.as_tensor(list(grades), dtype=x.dtype)
        if x.dim() != 2:
            raise ValueError(f"features must be 2-D; got {tuple(x.shape)}")
        if g.numel() != x.shape[0]:
            raise ValueError(
                f"grades has length {g.numel()}, features has {x.shape[0]} rows"
            )
        uniq = torch.unique(g)
        if uniq.numel() < 2:
            raise ValueError(
                "need at least two distinct grades to fit a discriminant; "
                f"got {uniq.tolist()}"
            )
        # Merge rare grades into the nearest populated one before splitting.
        counts = {float(u): int((g == u).sum()) for u in uniq}
        keep = sorted(v for v, c in counts.items() if c >= min_class_size)
        if len(keep) < 2:
            raise ValueError(
                f"only {len(keep)} grade(s) reach min_class_size="
                f"{min_class_size}; a discriminant needs at least two classes"
            )
        merged: List[float] = []
        remap = {u: u for u in counts}
        for grade, count in counts.items():
            if count >= min_class_size:
                continue
            nearest = min(keep, key=lambda k: (abs(k - grade), k))
            remap[grade] = nearest
            merged.append(grade)
        if merged:
            g = torch.tensor([remap[float(v)] for v in g], dtype=g.dtype)
            uniq = torch.tensor(sorted(set(remap.values())), dtype=g.dtype)

        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(x.shape[0], generator=gen)
        cut = int(round((1.0 - holdout_frac) * x.shape[0]))
        cut = max(1, min(x.shape[0] - 1, cut))
        tr_idx, te_idx = perm[:cut], perm[cut:]

        grades_tr = g[tr_idx]
        x_tr = x[tr_idx]
        d = x.shape[1]
        pooled = torch.zeros(d, d, dtype=x.dtype)
        dof = 0
        for u in uniq:
            block = x_tr[grades_tr == u]
            centred = block - block.mean(0)
            pooled += centred.T @ centred
            dof += block.shape[0] - 1
        sigma = pooled / dof
        mean_diag = float(torch.diagonal(sigma).mean())
        ridge = 1e-4 * mean_diag if shrinkage is None else float(shrinkage) * mean_diag
        if ridge > 0:
            sigma = sigma + ridge * torch.eye(d, dtype=x.dtype)

        eye = torch.eye(d, dtype=x.dtype)
        whitener = torch.stack(
            [solve_direction(sigma, eye[:, i], solver=solver) for i in range(d)],
            dim=1,
        )

        mu_w = torch.stack([x_tr[grades_tr == u].mean(0) for u in uniq]) @ whitener
        delta_proj = -0.5 * (mu_w * mu_w).sum(dim=1)

        # Rank selection. The between-class scatter of k classes has rank at
        # most k-1, and usually *less*: graded classes that lie on a line span
        # one direction no matter how many grades there are, and finite samples
        # give every class mean a small component in every other direction, so
        # the numerical rank is almost always k-1 even when the true layout is
        # one- or two-dimensional.
        #
        # Keeping all k-1 directions therefore fills the basis with noise, and
        # the subspace-overlap metric then measures the noise. The rule used
        # here keeps directions whose singular value is at least
        # `rank_tol_rel` times the largest one -- a *modelling choice*, not a
        # theorem, so it is a parameter and the spectrum is recorded in the
        # summary for inspection. Directions below it are discarded.
        centred = mu_w - mu_w.mean(0, keepdim=True)
        _u, svals, vh = torch.linalg.svd(centred, full_matrices=False)
        if svals.numel() and float(svals[0]) > 0:
            tol = float(svals[0]) * float(rank_tol_rel)
            numeric_rank = int((svals > tol).sum())
        else:
            numeric_rank = 0
        # The centring already produces the k-1, so nothing is subtracted here.
        rank = max(1, min(numeric_rank, centred.shape[0] - 1, vh.shape[0]))
        basis = vh[:rank].contiguous()
        spectrum = [float(v) for v in svals]

        obj = cls(
            name=name,
            mu_whitened=mu_w,
            delta_proj=delta_proj,
            basis=basis,
            thresholds=uniq,
            toxic_from=toxic_from,
            heldout_accuracy=0.0,
            heldout_side_accuracy=0.0,
            shrinkage=float(shrinkage or 0.0),
            whitener=whitener,
            embedding_dim=d,
            n_fit=int(tr_idx.numel()),
            spectrum=spectrum,
            merged_grades=tuple(merged),
        )
        obj.heldout_accuracy, obj.heldout_side_accuracy = obj.evaluate(
            x[te_idx], g[te_idx]
        )
        return obj

    # ------------------------------------------------------------- scoring

    @property
    def effective_rank(self) -> int:
        """Number of independent discriminant directions, ``k - 1``."""
        return int(self.basis.shape[0])

    @property
    def n_classes(self) -> int:
        """Number of distinct grades observed."""
        return int(self.mu_whitened.shape[0])

    def directions_matrix(self) -> torch.Tensor:
        """The ``(r, d)`` weight subspace.

        Returns:
            Rows span the discriminant subspace in the original feature space.
        """
        return self.basis

    def deltas(self, features: torch.Tensor) -> torch.Tensor:
        r"""Per-class discriminant scores.

        Args:
            features: Embeddings, shape ``(..., d)``.

        Returns:
            Tensor of shape ``(..., k)`` holding $\delta_j(x)$.
        """
        shape = features.shape[:-1]
        flat = features.reshape(-1, self.embedding_dim)
        out = flat @ self.whitener @ self.mu_whitened.T + self.delta_proj
        return out.reshape(*shape, self.n_classes)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        r"""The graded SASA margin, positive on the benign side.

        For two grades this reduces exactly to the binary LDA score
        $\delta_{\text{benign}} - \delta_{\text{toxic}}$.

        Args:
            features: Embeddings, shape ``(..., d)``.

        Returns:
            Tensor of shape ``(...)``.

        Raises:
            ValueError: If the observed grades leave one side of the
                toxic/non-toxic split empty.
        """
        delta = self.deltas(features)
        toxic = self.thresholds >= self.toxic_from
        benign = ~toxic
        if not bool(toxic.any()) or not bool(benign.any()):
            raise ValueError(
                f"attribute {self.name!r} has no examples on one side of "
                f"toxic_from={self.toxic_from}; grades were "
                f"{self.thresholds.tolist()}"
            )
        best_benign = delta[..., benign].max(dim=-1).values
        worst_toxic = delta[..., toxic].min(dim=-1).values
        return best_benign - worst_toxic

    def predicted_grade(self, features: torch.Tensor) -> torch.Tensor:
        """Predicted grade under the nearest-class-mean rule.

        The discriminant score $\\delta_j$ is a *distance*, not a logit, so
        the class prediction is its argmax -- testing the sign of the maximum
        would be a different and wrong rule.

        Args:
            features: Embeddings, shape ``(..., d)``.

        Returns:
            Tensor of shape ``(...)`` holding the predicted grade values.
        """
        return self.thresholds[self.deltas(features).argmax(dim=-1)]

    def side_prediction(self, features: torch.Tensor) -> torch.Tensor:
        """Binary side prediction, ``True`` on the benign side.

        Args:
            features: Embeddings, shape ``(..., d)``.

        Returns:
            Boolean tensor of shape ``(...)``.
        """
        return self.predicted_grade(features) < self.toxic_from

    def evaluate(
        self, features: torch.Tensor, grades: torch.Tensor
    ) -> Tuple[float, float]:
        """Grade accuracy and side accuracy on held-out data.

        Args:
            features: Held-out embeddings, shape ``(N, d)``.
            grades: Held-out grades, shape ``(N,)``.

        Returns:
            A ``(grade_accuracy, side_accuracy)`` tuple.
        """
        if features.shape[0] == 0:
            return 0.0, 0.0
        pred = self.predicted_grade(features)
        grade_hit = float((pred == grades).float().mean().item())
        side_acc = float(
            ((pred < self.toxic_from) == (grades < self.toxic_from))
            .float().mean().item()
        )
        return grade_hit, side_acc

    def token_bias(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        r"""Static vocabulary bias under the reference next-state estimator.

        Uses $\hat g_t = (g + e_t)/2$ and drops the constant the softmax
        annihilates, leaving a quantity that depends only on the token
        embeddings -- the collapse proved in :mod:`sasa.fast_margin`.

        Args:
            token_embeddings: Input-embedding matrix, shape ``(V, d)``.

        Returns:
            The steering vector at ``alpha = 1``, shape ``(V,)``.

        Raises:
            ValueError: If the embedding width does not match the fit.
        """
        if token_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"token embedding width {token_embeddings.shape[1]} "
                f"!= {self.embedding_dim}"
            )
        return 0.5 * self.score(token_embeddings)

    def passes_gate(self, threshold: float = IDENTIFIABILITY_GATE) -> bool:
        """Whether the subspace clears the pre-registered identifiability gate.

        Args:
            threshold: Minimum held-out side accuracy.

        Returns:
            ``True`` if the attribute is identifiable enough to study.
        """
        return self.heldout_side_accuracy >= threshold

    def summary(self) -> Dict[str, object]:
        """Return scalar diagnostics as a plain dictionary."""
        return {
            "name": self.name,
            "n_classes": self.n_classes,
            "effective_rank": self.effective_rank,
            "n_fit": self.n_fit,
            "heldout_grade_accuracy": self.heldout_accuracy,
            "heldout_side_accuracy": self.heldout_side_accuracy,
            "shrinkage": self.shrinkage,
            "passes_gate": self.passes_gate(),
            "between_class_spectrum": [round(float(v), 6)
                                       for v in self.spectrum],
            "merged_grades": [float(v) for v in self.merged_grades],
        }


class AttributeSet:
    """A collection of attribute subspaces and their interaction geometry.

    Attributes:
        attributes: The fitted subspaces, in insertion order.
    """

    def __init__(self, attributes: Optional[Sequence[GradedSubspace]] = None) -> None:
        """Initialise the set.

        Args:
            attributes: Optional initial subspaces.
        """
        self.attributes: List[GradedSubspace] = list(attributes or [])

    def add(self, subspace: GradedSubspace) -> "AttributeSet":
        """Add a subspace, returning self for chaining.

        Args:
            subspace: The subspace to add.

        Returns:
            ``self``.
        """
        self.attributes.append(subspace)
        return self

    def __len__(self) -> int:
        return len(self.attributes)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.attributes[key]
        for attr in self.attributes:
            if attr.name == key:
                return attr
        raise KeyError(f"no attribute named {key!r}")

    @property
    def names(self) -> List[str]:
        """Names of the contained attributes."""
        return [a.name for a in self.attributes]

    def identifiable(self, threshold: float = IDENTIFIABILITY_GATE) -> "AttributeSet":
        """Return the subset that clears the identifiability gate.

        Args:
            threshold: Minimum held-out side accuracy.

        Returns:
            A new :class:`AttributeSet` containing only gated-in attributes.
        """
        return AttributeSet(
            [a for a in self.attributes if a.passes_gate(threshold)]
        )

    # ----------------------------------------------------------- geometry

    def principal_angles(
        self, a: GradedSubspace, b: GradedSubspace
    ) -> torch.Tensor:
        """Principal angles between two weight subspaces.

        Args:
            a: First subspace.
            b: Second subspace.

        Returns:
            Tensor of singular values of ``A B^T`` in ``[0, 1]``, where each is
            the cosine of a principal angle.
        """
        if a.embedding_dim != b.embedding_dim:
            raise ValueError(
                "subspaces must share a feature width; "
                f"{a.embedding_dim} vs {b.embedding_dim}"
            )
        return torch.linalg.svdvals(a.directions_matrix() @ b.directions_matrix().T)

    def interference(self) -> List["InterferencePair"]:
        """Static interference prediction for every unordered attribute pair.

        Returns:
            One :class:`InterferencePair` per pair, ordered by descending
            overlap so the worst offenders come first.
        """
        pairs: List[InterferencePair] = []
        for a, b in combinations(self.attributes, 2):
            cos = self.principal_angles(a, b)
            r = max(1, int(cos.numel()))
            overlap = float((cos ** 2).sum() / r)
            pairs.append(InterferencePair(
                attribute_a=a.name,
                attribute_b=b.name,
                subspace_overlap=overlap,
                max_cosine=float(cos.max()),
                min_principal_angle=float(
                    torch.arccos(cos.clamp(max=1.0)).max()
                ),
                rank_a=a.effective_rank,
                rank_b=b.effective_rank,
            ))
        pairs.sort(key=lambda p: -p.subspace_overlap)
        return pairs

    def compose_logits(
        self,
        logits: torch.Tensor,
        token_embeddings: torch.Tensor,
        alphas: Dict[str, float],
        normalise: bool = False,
    ) -> torch.Tensor:
        """Compose several attribute biases into one adjusted logit vector.

        Args:
            logits: Model logits, shape ``(V,)`` or ``(B, V)``.
            token_embeddings: Input-embedding matrix, shape ``(V, d)``.
            alphas: Per-attribute strength. Attributes with no entry are ignored.
            normalise: If True, divide each attribute's bias by its own standard
                deviation before weighting, so that alphas mean the same thing
                across attributes of different scale. This is the same
                scale-free discipline as the KL-budget calibration in
                Appendix C.4.

        Returns:
            Adjusted logits with the same shape as ``logits``.

        Raises:
            KeyError: If ``alphas`` names an attribute that is not present.
        """
        unknown = set(alphas) - set(self.names)
        if unknown:
            raise KeyError(f"unknown attributes in alphas: {sorted(unknown)}")
        out = logits
        for attr in self.attributes:
            alpha = alphas.get(attr.name, 0.0)
            if alpha == 0.0:
                continue
            bias = attr.token_bias(token_embeddings)
            if normalise:
                scale = bias.std()
                if scale > 0:
                    bias = bias / scale
            out = out + alpha * bias
        return out

    def single_attribute_score(
        self,
        features: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """Score features with one attribute's margin.

        Args:
            features: Embeddings, shape ``(..., d)``.
            name: Attribute name.

        Returns:
            The margin, shape ``(...)``.
        """
        return self[name].score(features)


@dataclass
class InterferencePair:
    """Static interference prediction for one attribute pair.

    Attributes:
        attribute_a: First attribute name.
        attribute_b: Second attribute name.
        subspace_overlap: Mean squared principal cosine in ``[0, 1]``; equals
            the squared cosine when both subspaces are rank one.
        max_cosine: Largest principal cosine.
        min_principal_angle: Largest principal angle, in radians. Near
            $\\pi/2$ means the margins barely overlap and can be maximised
            simultaneously.
        rank_a: Effective rank of the first subspace.
        rank_b: Effective rank of the second subspace.
    """

    attribute_a: str
    attribute_b: str
    subspace_overlap: float
    max_cosine: float
    min_principal_angle: float
    rank_a: int
    rank_b: int

    def as_dict(self) -> Dict[str, object]:
        """Return the pair as a plain dictionary."""
        return {
            "attribute_a": self.attribute_a,
            "attribute_b": self.attribute_b,
            "subspace_overlap": self.subspace_overlap,
            "max_cosine": self.max_cosine,
            "min_principal_angle": self.min_principal_angle,
            "rank_a": self.rank_a,
            "rank_b": self.rank_b,
        }
