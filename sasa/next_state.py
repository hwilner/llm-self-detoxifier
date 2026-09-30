"""NLMA: a non-linear next-state margin approximation.

Why this module exists
----------------------
Under the reference next-state estimator ``g_hat = (g(c) + e_t)/2`` the SASA
margin is provably independent of the context, because every context-dependent
term is a constant shift that the softmax cancels (see
:mod:`sasa.fast_margin`). The same is true of *any* affine estimator
``A g + B e_t``: the ``A g`` contribution is again constant across candidate
tokens.

Restoring context-adaptive margin steering therefore requires an estimator that
couples context and candidate **multiplicatively**. NLMA uses the cheapest
family that does:

.. math::
    \\hat g^{\\text{NLMA}}_t
      = A\\,g + B\\,e_t + \\sum_{r=1}^{R} (u_r^\\top g)\\,(v_r^\\top e_t)\\,r_r .

The margin then acquires a term that survives the softmax,

.. math::
    \\mathrm{margin}_t \\supset
      \\frac{\\sum_r (u_r^\\top g)\\,(v_r^\\top e_t)\\,(r_r^\\top w)}
           {\\lVert w \\rVert},

which is a genuine function of both the context and the candidate token.

Cost
----
Precompute ``M = E V^\\top \\in \\mathbb{R}^{V \\times R}`` once. A decoding step
then costs ``O(Rd + VR)`` instead of the reference implementation's ``O(Vd)``.
For Llama-3.1-8B with ``R = 8`` that is ~1.0 M multiply-adds versus 525 M.

Fitting
-------
The affine part ``(A, B)`` is fitted in closed form by ridge regression. Only
the low-rank interaction ``(U, V, R)`` is fitted by Adam, on frozen features
collected from forward passes. No gradient is ever propagated through the
language model, and the model itself is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

__all__ = [
    "NextStateModel",
    "collect_next_state_pairs",
    "fit_next_state_model",
    "MarginStatistics",
    "margin_statistics",
]


@dataclass
class NextStateModel:
    """A fitted low-rank non-linear next-state estimator.

    The affine part is parameterised with centred features and an explicit
    intercept::

        mean + (g - context_mean) @ a_mat.T + (e - token_mean) @ b_mat.T

    so that ``mean`` is the only offset and is never counted twice. The
    interaction term is evaluated on *raw* features.

    Attributes:
        mean: Intercept of the affine part, shape ``(d,)``.
        a_mat: Context matrix of the affine part, shape ``(d, d)``.
        b_mat: Token matrix of the affine part, shape ``(d, d)``.
        context_mean: Mean context used to centre ``g``, shape ``(d,)``.
        token_mean: Mean token embedding used to centre ``e``, shape ``(d,)``.
        u: Context factors of the interaction, shape ``(R, d)``.
        v: Token factors of the interaction, shape ``(R, d)``.
        r: Output directions of the interaction, shape ``(R, d)``.
        token_projection: Precomputed ``E V^T``, shape ``(V, R)``.
        fit_stats: Diagnostics recorded at fit time.
    """

    mean: torch.Tensor
    a_mat: torch.Tensor
    b_mat: torch.Tensor
    context_mean: torch.Tensor
    token_mean: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    r: torch.Tensor
    token_projection: torch.Tensor
    fit_stats: dict

    @property
    def rank(self) -> int:
        """Interaction rank ``R``."""
        return int(self.u.shape[0])

    @property
    def embedding_dim(self) -> int:
        """Hidden size ``d``."""
        return int(self.mean.shape[0])

    def affine_next_states(
        self, context_hidden: torch.Tensor, token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """The affine part of the estimate, without the interaction.

        Useful as a baseline: this is the best possible member of the affine
        family, and by Theorem 3 it must remain context-independent *after* the
        softmax. Any measured benefit of the full model therefore comes from
        the interaction term, not from a better affine fit.

        Args:
            context_hidden: Context hidden state, shape ``(d,)``.
            token_embeddings: Input-embedding matrix, shape ``(V, d)``.

        Returns:
            Tensor of shape ``(V, d)``.
        """
        return (
            self.mean.unsqueeze(0)
            + (context_hidden.unsqueeze(0) - self.context_mean) @ self.a_mat.T
            + (token_embeddings - self.token_mean) @ self.b_mat.T
        )

    def interaction(
        self, context_hidden: torch.Tensor, token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """The low-rank bilinear interaction term alone.

        Args:
            context_hidden: Context hidden state, shape ``(d,)``.
            token_embeddings: Input-embedding matrix, shape ``(V, d)``.

        Returns:
            Tensor of shape ``(V, d)``, zero if the model has rank zero.
        """
        if self.rank == 0:
            return torch.zeros(token_embeddings.shape[0], self.embedding_dim)
        context_factors = self.u @ context_hidden          # (R,)
        token_factors = token_embeddings @ self.v.T       # (V, R)
        # (V, R) * (R,) -> (V, R), then combine the R output directions.
        return (token_factors * context_factors.unsqueeze(0)) @ self.r

    def estimate(
        self, context_hidden: torch.Tensor, token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Estimate the next hidden state for every candidate token.

        Args:
            context_hidden: Context hidden state, shape ``(d,)``.
            token_embeddings: Input-embedding matrix, shape ``(V, d)``.

        Returns:
            Estimated next states, shape ``(V, d)``.

        Raises:
            ValueError: If the context or embedding widths disagree with the
                fitted dimension.
        """
        if context_hidden.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"context width {context_hidden.shape[-1]} does not match "
                f"fitted dimension {self.embedding_dim}"
            )
        if token_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"token embedding width {token_embeddings.shape[1]} does not "
                f"match fitted dimension {self.embedding_dim}"
            )
        return (
            self.affine_next_states(context_hidden, token_embeddings)
            + self.interaction(context_hidden, token_embeddings)
        )

    def predict_pairs(
        self,
        contexts: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next state for paired ``(context, token)`` observations.

        This is the per-observation counterpart of :meth:`estimate` and is the
        right entry point for evaluating fit quality, where each row has its own
        candidate token.

        Args:
            contexts: Context hidden states, shape ``(N, d)``.
            tokens: Candidate token embeddings, shape ``(N, d)``.

        Returns:
            Predicted next states, shape ``(N, d)``.

        Raises:
            ValueError: If the two inputs disagree on sample count.
        """
        if contexts.shape != tokens.shape:
            raise ValueError(
                f"shape mismatch: {tuple(contexts.shape)} vs {tuple(tokens.shape)}"
            )
        out = (
            self.mean
            + (contexts - self.context_mean) @ self.a_mat.T
            + (tokens - self.token_mean) @ self.b_mat.T
        )
        if self.rank:
            out = out + ((contexts @ self.u.T) * (tokens @ self.v.T)) @ self.r
        return out

    def margins(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        context_hidden: torch.Tensor,
        token_embeddings: torch.Tensor,
        affine_only: bool = False,
    ) -> torch.Tensor:
        """Per-token margins under the fitted estimator.

        Args:
            weight: Subspace direction ``w``, shape ``(d,)``.
            bias: Subspace bias ``b``, shape ``(d,)``.
            context_hidden: Context hidden state, shape ``(d,)``.
            token_embeddings: Input-embedding matrix, shape ``(V, d)``.
            affine_only: If True, use only the affine part, isolating the
                effect of the interaction term.

        Returns:
            Margins, shape ``(V,)``.
        """
        w_norm = torch.norm(weight)
        if affine_only:
            states = self.affine_next_states(context_hidden, token_embeddings)
        else:
            states = self.estimate(context_hidden, token_embeddings)
        return (states - bias.unsqueeze(0)) @ weight / w_norm

    def context_sensitivity(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        context_a: torch.Tensor,
        context_b: torch.Tensor,
        token_embeddings: torch.Tensor,
        affine_only: bool = False,
    ) -> float:
        """Total-variation distance between the margin distributions of two contexts.

        This is the operational test of Theorem 3: the affine variant must
        return ~0, the full model must return a strictly positive value.

        Args:
            weight: Subspace direction ``w``, shape ``(d,)``.
            bias: Subspace bias ``b``, shape ``(d,)``.
            context_a: First context hidden state, shape ``(d,)``.
            context_b: Second context hidden state, shape ``(d,)``.
            token_embeddings: Input-embedding matrix, shape ``(V, d)``.
            affine_only: Restrict the estimator to its affine part.

        Returns:
            Total-variation distance between the two softmax margin vectors.
        """
        ma = self.margins(weight, bias, context_a, token_embeddings, affine_only)
        mb = self.margins(weight, bias, context_b, token_embeddings, affine_only)
        pa = torch.softmax(ma, dim=-1)
        pb = torch.softmax(mb, dim=-1)
        return float(0.5 * torch.abs(pa - pb).sum().item())


def collect_next_state_pairs(
    model,
    tokenizer,
    prompts: List[str],
    layer: int,
    device: torch.device,
    max_candidates: int = 24,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect ``(context, candidate embedding, true next state)`` triples.

    For each prompt the model's true next hidden state is obtained by running an
    actual forward pass with the candidate token appended. This is the ground
    truth that any estimator -- affine or not -- is trying to approximate.

    Args:
        model: A causal language model.
        tokenizer: The matching tokenizer.
        prompts: Context strings to expand.
        layer: Index into ``output.hidden_states`` to read.
        device: Device to run on.
        max_candidates: Candidate tokens probed per prompt. Chosen with a
            fixed generator seed so collection is deterministic.
        seed: RNG seed for candidate selection.

    Returns:
        A ``(contexts, token_embeds, targets)`` triple with shapes
        ``(N, d)``, ``(N, d)`` and ``(N, d)``.

    Raises:
        ValueError: If ``prompts`` is empty.
    """
    if not prompts:
        raise ValueError("prompts must not be empty")

    gen = torch.Generator().manual_seed(seed)
    contexts: List[torch.Tensor] = []
    tokens: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            base = model(ids, output_hidden_states=True)
            g = base.hidden_states[layer][0, -1, :].cpu()
            logits = base.logits[0, -1, :]
            k = min(max_candidates, int(logits.shape[0]))
            idx = torch.multinomial(
                torch.softmax(logits, dim=-1), k, generator=gen
            ).tolist()

            emb = model.get_input_embeddings().weight
            for t in idx:
                extended = torch.cat(
                    [ids, torch.tensor([[t]], device=device)], dim=1
                )
                out = model(extended, output_hidden_states=True)
                nxt = out.hidden_states[layer][0, -1, :].cpu()
                contexts.append(g)
                tokens.append(emb[t].detach().cpu())
                targets.append(nxt)

    return torch.stack(contexts), torch.stack(tokens), torch.stack(targets)


def _fit_affine(
    contexts: torch.Tensor,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    ridge: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Ridge-fit ``mean + A (g - g_bar) + B (e - e_bar)`` in closed form.

    Features are centred and an explicit intercept column is included, so the
    returned ``mean`` is the sole offset.

    Args:
        contexts: Context states, shape ``(N, d)``.
        tokens: Candidate embeddings, shape ``(N, d)``.
        targets: True next states, shape ``(N, d)``.
        ridge: L2 penalty.

    Returns:
        A ``(mean, A, B, g_bar, e_bar, residual)`` tuple; ``A`` and ``B`` have
        shape ``(d, d)`` and ``residual`` is ``targets - affine prediction``.
    """
    n, d = contexts.shape
    g_bar = contexts.mean(0)
    e_bar = tokens.mean(0)
    design = torch.cat(
        [
            torch.ones(n, 1, dtype=targets.dtype),
            contexts - g_bar,
            tokens - e_bar,
        ],
        dim=1,
    )
    gram = design.T @ design + ridge * torch.eye(2 * d + 1, dtype=targets.dtype)
    coef = torch.linalg.solve(gram, design.T @ targets)
    mean = coef[0]                       # (d,) intercept
    # Stored transposed so that reconstruction is ``x @ a_mat.T``:
    #   (x @ a_mat.T)[n, k] = sum_j x[n, j] * coef[1 + j, k]
    a_mat = coef[1:1 + d].transpose(0, 1).contiguous()   # (d, d)
    b_mat = coef[1 + d:].transpose(0, 1).contiguous()     # (d, d)
    residual = targets - design @ coef
    return mean, a_mat, b_mat, g_bar, e_bar, residual


def fit_next_state_model(
    contexts: torch.Tensor,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    token_embeddings: torch.Tensor,
    rank: int = 4,
    ridge: float = 1e-2,
    steps: int = 400,
    lr: float = 3e-3,
    batch_size: int = 256,
    seed: int = 0,
    verbose: bool = False,
) -> NextStateModel:
    """Fit an NLMA model to collected next-state triples.

    The affine part is solved in closed form; only the interaction factors are
    optimised, with Adam, on frozen features. Gradients never pass through the
    language model.

    Args:
        contexts: Context hidden states, shape ``(N, d)``.
        tokens: Candidate token embeddings, shape ``(N, d)``.
        targets: True next hidden states, shape ``(N, d)``.
        token_embeddings: The full embedding table, shape ``(V, d)``, used to
            precompute the token projection.
        rank: Interaction rank ``R``. ``0`` yields the affine model.
        ridge: Ridge penalty for the affine solve.
        steps: Adam steps for the interaction.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        seed: RNG seed for mini-batch sampling.
        verbose: Print the loss every 50 steps.

    Returns:
        A fitted :class:`NextStateModel`.

    Raises:
        ValueError: If the three inputs disagree on sample count or width.
    """
    if not (contexts.shape == tokens.shape == targets.shape):
        raise ValueError(
            f"shape mismatch: {tuple(contexts.shape)}, {tuple(tokens.shape)}, "
            f"{tuple(targets.shape)}"
        )
    d = contexts.shape[1]
    if token_embeddings.shape[1] != d:
        raise ValueError(
            f"token_embeddings width {token_embeddings.shape[1]} != {d}"
        )

    mean, a_mat, b_mat, g_bar, e_bar, residual = _fit_affine(
        contexts, tokens, targets, ridge
    )
    base_mse = float(residual.pow(2).mean().item())

    gen = torch.Generator().manual_seed(seed)
    scale = 1.0 / (d ** 0.5)
    u = torch.randn(rank, d, generator=gen) * scale
    v = torch.randn(rank, d, generator=gen) * scale
    r_mat = torch.zeros(rank, d)

    if rank > 0:
        u.requires_grad_(True)
        v.requires_grad_(True)
        r_mat.requires_grad_(True)
        opt = torch.optim.Adam([u, v, r_mat], lr=lr)
        n = contexts.shape[0]
        for step in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),), generator=gen)
            gc, gt, gr = contexts[idx], tokens[idx], residual[idx]
            pred = ((gc @ u.T) * (gt @ v.T)) @ r_mat
            loss = (pred - gr).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if verbose and step % 50 == 0:
                print(f"      [nlma] step {step:4d} mse={float(loss):.6f}")
        u, v, r_mat = u.detach(), v.detach(), r_mat.detach()

    with torch.no_grad():
        token_projection = token_embeddings @ v.T  # (V, R)

    return NextStateModel(
        mean=mean,
        a_mat=a_mat,
        b_mat=b_mat,
        context_mean=g_bar,
        token_mean=e_bar,
        u=u,
        v=v,
        r=r_mat,
        token_projection=token_projection,
        fit_stats={
            "rank": rank,
            "n_samples": int(contexts.shape[0]),
            "ridge": ridge,
            "steps": steps,
            "lr": lr,
            "affine_mse": base_mse,
            "final_interaction_mse": float(
                (((contexts @ u.T) * (tokens @ v.T)) @ r_mat - residual)
                .pow(2).mean().item()
            ) if rank > 0 else 0.0,
        },
    )


@dataclass
class MarginStatistics:
    """How strongly the margin vector responds to a change of context.

    Attributes:
        total_variation: TV distance between the two margin softmaxes.
        spearman: Rank correlation between the two margin vectors.
        pearson: Pearson correlation between the two margin vectors.
        cosine: Cosine similarity between the two margin vectors.
        top1_agreement: Fraction of tokens in the top-1 % whose rank is shared.
    """

    total_variation: float
    spearman: float
    pearson: float
    cosine: float
    top1_agreement: float

    def as_dict(self) -> dict:
        """Return the statistics as a plain dictionary."""
        return {
            "total_variation": self.total_variation,
            "spearman": self.spearman,
            "pearson": self.pearson,
            "cosine": self.cosine,
            "top1_agreement": self.top1_agreement,
        }


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    """Average-rank transform with tie handling.

    Args:
        x: A 1-D tensor.

    Returns:
        Tensor of average ranks, same shape.
    """
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(1, x.numel() + 1, dtype=x.dtype)
    # Average ties.
    sorted_x = x[order]
    i = 0
    while i < x.numel():
        j = i
        while j + 1 < x.numel() and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def margin_statistics(margin_a: torch.Tensor, margin_b: torch.Tensor) -> MarginStatistics:
    """Compare two per-token margin vectors.

    Args:
        margin_a: Margins from the first context, shape ``(V,)``.
        margin_b: Margins from the second context, shape ``(V,)``.

    Returns:
        A :class:`MarginStatistics` record.

    Raises:
        ValueError: If the two vectors differ in length.
    """
    if margin_a.shape != margin_b.shape:
        raise ValueError(f"shape mismatch: {margin_a.shape} vs {margin_b.shape}")

    pa = torch.softmax(margin_a, dim=-1)
    pb = torch.softmax(margin_b, dim=-1)
    tv = float(0.5 * torch.abs(pa - pb).sum().item())

    ra, rb = _rankdata(margin_a.double()), _rankdata(margin_b.double())
    spearman = float(
        torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    ) if ra.std() > 0 and rb.std() > 0 else 0.0
    pearson = float(
        torch.corrcoef(torch.stack([margin_a.double(), margin_b.double()]))[0, 1].item()
    )
    na, nb = torch.norm(margin_a), torch.norm(margin_b)
    cosine = float((margin_a @ margin_b / (na * nb)).item()) if na > 0 and nb > 0 else 0.0

    k = max(1, margin_a.numel() // 100)
    top_a = set(torch.topk(margin_a, k).indices.tolist())
    top_b = set(torch.topk(margin_b, k).indices.tolist())
    return MarginStatistics(
        total_variation=tv,
        spearman=spearman,
        pearson=pearson,
        cosine=cosine,
        top1_agreement=len(top_a & top_b) / k,
    )
