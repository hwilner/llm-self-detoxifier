"""Cached and batched decoding, with an exactness guarantee.

The cost that actually dominates
-------------------------------
`sasa.sampler.SASASampler.generate` re-runs a full forward pass over the whole
prefix at every step and never populates a key-value cache, so generating ``L``
tokens costs ``O(L^2)`` forward work in the context length. Appendix B removes
the per-token margin cost exactly; this module removes the *other* cost, which
is the one that dominates in practice and the one no efficiency claim about
SASA can be stated without addressing.

What is guaranteed
------------------
Every function here is checked against the uncached, unbatched reference for
distributional equality, not merely for plausibility:

* :func:`cached_generate` with a fixed RNG seed reproduces the same token
  sequence as an uncached loop, because the cached and uncached logits agree to
  float32 precision at every step (``test_cached_matches_uncached_logits``).
* :func:`batched_sample` applies the same static bias independently to each
  row, so a batch of size 1 reproduces the unbatched distribution exactly and a
  batch of size ``B`` reproduces ``B`` independent single-row draws.

The static bias makes batching unusually cheap here: because the steering vector
is context-free (Theorem 1 in :mod:`sasa.fast_margin`), one ``(V,)`` add serves
every row of the batch. A context-dependent margin would have to be computed
per row, which is the main reason batching a steered decoder is otherwise
awkward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "DecodeOutput",
    "_check_bias",
    "apply_top_k",
    "apply_top_p",
    "sample_from_logits",
    "cached_generate",
    "uncached_generate",
    "batched_sample",
    "token_budget",
]


@dataclass
class DecodeOutput:
    """Result of a decode loop.

    Attributes:
        tokens: Generated token ids, shape ``(L,)``.
        step_logits_norm: L2 norm of the *unsteered* logits at each step,
            length ``L``. Useful as a cheap determinism fingerprint: two runs
            that agree on this agree on the model's own computation.
        prefill_tokens: Number of tokens in the prompt.
    """

    tokens: torch.Tensor
    step_logits_norm: torch.Tensor
    prefill_tokens: int


def _check_bias(bias, vocab: int) -> None:
    """Validate a steering bias against the model's vocabulary size.

    A bias may be a single ``(V,)`` vector, applied at every step, or a
    ``(L, V)`` stack giving a per-step schedule, which is how
    :mod:`sasa.scheduling` reaches the decoder. Only the *magnitude* can vary
    under the static collapse (Theorem 1 in :mod:`sasa.fast_margin`), so a
    schedule is a rank-one rescaling of one precomputed vector -- no extra
    per-token cost.

    Args:
        bias: The bias tensor, or ``None``.
        vocab: Model vocabulary size.

    Raises:
        ValueError: If the bias is present with an unexpected shape.
    """
    if bias is None:
        return
    if bias.dim() == 1:
        if bias.shape[0] != vocab:
            raise ValueError(f"bias has V={bias.shape[0]}, model has V={vocab}")
    elif bias.dim() == 2:
        if bias.shape[1] != vocab:
            raise ValueError(
                f"bias schedule has V={bias.shape[1]}, model has V={vocab}"
            )
    else:
        raise ValueError(
            f"bias must be (V,) or (L, V); got {tuple(bias.shape)}"
        )


def _step_bias(bias, step: int) -> Optional[torch.Tensor]:
    """Select the bias for one decoding step.

    Args:
        bias: A ``(V,)`` or ``(L, V)`` bias, or ``None``.
        step: Zero-based step index.

    Returns:
        The bias for this step, or ``None``. A per-step stack is clamped at its
        last row once generation runs past the scheduled length.
    """
    if bias is None:
        return None
    if bias.dim() == 1:
        return bias
    return bias[min(step, bias.shape[0] - 1)]


def apply_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    """Mask all but the ``top_k`` highest entries.

    Args:
        logits: Logits of shape ``(..., V)``.
        top_k: Keep count, or ``None`` to skip.

    Returns:
        A tensor with the same shape as ``logits``.

    Raises:
        ValueError: If ``top_k`` is not positive.
    """
    if top_k is None:
        return logits
    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}")
    k = min(top_k, logits.shape[-1])
    threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    """Apply nucleus filtering.

    Args:
        logits: Logits of shape ``(..., V)``.
        top_p: Cumulative probability threshold, or ``None`` to skip.

    Returns:
        A tensor with the same shape as ``logits``.

    Raises:
        ValueError: If ``top_p`` is outside ``(0, 1]``.
    """
    if top_p is None:
        return logits
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must lie in (0, 1]; got {top_p}")
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    drop = cumulative > top_p
    drop[..., 0] = False  # never empty the support
    drop = drop.scatter(-1, sorted_idx, drop)
    return logits.masked_fill(drop, float("-inf"))


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample indices from a filtered logit tensor.

    Args:
        logits: Logits of shape ``(..., V)``.
        temperature: Divisor applied before filtering.
        top_k: Top-k truncation, or ``None``.
        top_p: Nucleus threshold, or ``None``.
        generator: Optional RNG for reproducibility.

    Returns:
        Sampled indices of shape ``(...)``, dtype ``int64``.

    Raises:
        ValueError: If ``temperature`` is non-positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive; got {temperature}")
    scaled = logits / temperature
    scaled = apply_top_k(scaled, top_k)
    scaled = apply_top_p(scaled, top_p)
    probs = F.softmax(scaled, dim=-1)
    if generator is None:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def uncached_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    bias: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    eos_token_id: Optional[int] = None,
) -> DecodeOutput:
    """Reference decode loop with no key-value cache.

    Kept as the ground truth that :func:`cached_generate` is checked against.

    Args:
        model: A causal language model.
        input_ids: Prompt ids of shape ``(1, P)``.
        max_new_tokens: Tokens to generate.
        bias: Optional static steering vector of shape ``(V,)``.
        temperature: Sampling temperature.
        top_k: Top-k truncation, or ``None``.
        top_p: Nucleus threshold, or ``None``.
        generator: Optional RNG for reproducibility.
        eos_token_id: Stop token, or ``None`` to never stop early.

    Returns:
        A :class:`DecodeOutput`.

    Raises:
        ValueError: If ``max_new_tokens`` is negative or ``bias`` has the wrong
            width.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0; got {max_new_tokens}")
    ids = input_ids
    produced: List[int] = []
    norms: List[float] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(ids)
            logits = out.logits[0, -1, :]
            norms.append(float(logits.norm().item()))
            if bias is not None:
                _check_bias(bias, logits.shape[0])
                logits = logits + bias
            token = int(sample_from_logits(
                logits.unsqueeze(0), temperature, top_k, top_p, generator
            ).item())
            if eos_token_id is not None and token == eos_token_id:
                break
            produced.append(token)
            ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], dim=1)
    return DecodeOutput(
        tokens=torch.tensor(produced, dtype=torch.long),
        step_logits_norm=torch.tensor(norms, dtype=torch.float32),
        prefill_tokens=int(input_ids.shape[1]),
    )


def cached_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    bias: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    eos_token_id: Optional[int] = None,
) -> DecodeOutput:
    """Decode with a populated key-value cache, giving ``O(L)`` forward work.

    Args:
        model: A causal language model.
        input_ids: Prompt ids of shape ``(1, P)``.
        max_new_tokens: Tokens to generate.
        bias: Steering vector, shape ``(V,)`` for a constant strength, or
            ``(L, V)`` for a per-step schedule.
        temperature: Sampling temperature.
        top_k: Top-k truncation, or ``None``.
        top_p: Nucleus threshold, or ``None``.
        generator: Optional RNG for reproducibility.
        eos_token_id: Stop token, or ``None`` to never stop early.

    Returns:
        A :class:`DecodeOutput`, numerically identical to
        :func:`uncached_generate` for the same seed.

    Raises:
        ValueError: If ``max_new_tokens`` is negative, or ``bias`` has the wrong
            shape.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0; got {max_new_tokens}")
    produced: List[int] = []
    norms: List[float] = []
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :]
        _check_bias(bias, int(logits.shape[0]))
        for step in range(max_new_tokens):
            norms.append(float(logits.norm().item()))
            step_bias = _step_bias(bias, step)
            steered = logits if step_bias is None else logits + step_bias
            token = int(sample_from_logits(
                steered.unsqueeze(0), temperature, top_k, top_p, generator
            ).item())
            if eos_token_id is not None and token == eos_token_id:
                break
            produced.append(token)
            step = torch.tensor([[token]], device=input_ids.device)
            out = model(step, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :]
    return DecodeOutput(
        tokens=torch.tensor(produced, dtype=torch.long),
        step_logits_norm=torch.tensor(norms, dtype=torch.float32),
        prefill_tokens=int(input_ids.shape[1]),
    )


def batched_sample(
    logits: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generators: Optional[Sequence[torch.Generator]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one token per row, applying the same static bias to every row.

    Because the steering vector is context-free, one ``(V,)`` addition serves
    the whole batch, and the whole batch can be sampled with a single
    ``torch.multinomial`` call over a ``(B, V)`` distribution. That is the fast
    path, and it is what makes batching worth doing: a per-row Python loop costs
    exactly as much per item at batch 32 as at batch 1, which is no batching at
    all.

    Two reproducibility contracts are offered, and they differ:

    * ``generators`` (one per row) gives a draw that is **independent of the
      batch size** -- the same row with the same generator yields the same token
      whatever else is in the batch -- at the cost of a Python loop.
    * ``generator`` (one for the batch) is a single vectorised call and is fast,
      but a row's token depends on its position in the batch.

    Args:
        logits: Unsteered logits of shape ``(B, V)``.
        bias: Optional static steering vector of shape ``(V,)``.
        temperature: Sampling temperature.
        top_k: Top-k truncation, or ``None``.
        top_p: Nucleus threshold, or ``None``.
        generators: One generator per row, or ``None``.
        generator: A single generator for the whole batch, or ``None``.

    Returns:
        Sampled indices of shape ``(B,)``, dtype ``int64``.

    Raises:
        ValueError: If ``logits`` is not 2-D, ``bias`` has the wrong width, both
            generator styles are supplied, or the number of per-row generators
            does not match the batch size.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V); got {tuple(logits.shape)}")
    if bias is not None and bias.shape[0] != logits.shape[1]:
        raise ValueError(
            f"bias has V={bias.shape[0]}, logits have V={logits.shape[1]}"
        )
    if generators is not None and generator is not None:
        raise ValueError(
            "supply either per-row 'generators' or a single 'generator', "
            "not both"
        )
    if generators is not None and len(generators) != logits.shape[0]:
        raise ValueError(
            f"expected {logits.shape[0]} generators, got {len(generators)}"
        )

    adjusted = logits if bias is None else logits + bias

    if generators is None:
        # Vectorised: one multinomial over the whole batch.
        scaled = adjusted / temperature
        scaled = apply_top_k(scaled, top_k)
        scaled = apply_top_p(scaled, top_p)
        probs = F.softmax(scaled, dim=-1)
        if generator is not None:
            return torch.multinomial(
                probs, num_samples=1, generator=generator).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    out = []
    for row in range(logits.shape[0]):
        row_logits = adjusted[row]
        out.append(int(sample_from_logits(
            row_logits, temperature, top_k, top_p, generators[row]
        ).item()))
    return torch.tensor(out, dtype=torch.long)


def token_budget(
    prompt_len: int,
    n_new_tokens: int,
    n_layers: int = 0,
) -> Dict[str, int]:
    """Count forwarded token positions with and without a cache.

    The uncached loop re-processes the entire prefix at every step, so its work
    grows quadratically in the number of generated tokens. This is the quantity
    every efficiency claim about SASA should be stated against.

    Args:
        prompt_len: Prompt length in tokens.
        n_new_tokens: Tokens to generate.
        n_layers: Number of transformer layers, recorded for context.

    Returns:
        A dict with ``uncached``, ``cached`` and ``speedup`` counts, and the
        inputs echoed back.
    """
    if prompt_len < 0 or n_new_tokens < 0:
        raise ValueError("prompt_len and n_new_tokens must be non-negative")
    uncached = sum(prompt_len + i for i in range(1, n_new_tokens + 1))
    cached = prompt_len + n_new_tokens
    return {
        "uncached_token_positions": uncached,
        "cached_token_positions": cached,
        "speedup": uncached / cached if cached else 0.0,
        "prompt_len": prompt_len,
        "n_new_tokens": n_new_tokens,
        "n_layers": n_layers,
    }
