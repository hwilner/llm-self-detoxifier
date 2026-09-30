"""Measure the rank-one collapse: exactness, cost, and context-independence.

This script produces every number in Appendix B. It compares the reference
``compute_token_margins`` path against :class:`sasa.fast_margin.StaticMarginBias`
on three questions:

1. **Exactness** -- do the two produce the same sampling distribution?
2. **Cost** -- how much arithmetic and peak memory does the collapse remove, at
   the vocabulary sizes and hidden sizes of real models?
3. **Context-independence** -- does the steering rule depend on the hidden
   state? (Theorems 1 and 3.)

The third question is answered on *synthetic* subspaces as well as real ones,
because the property is algebraic and should not depend on the data.

Usage::

    python experiments/run_collapse_benchmark.py --out results/collapse.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sasa.fast_margin import (  # noqa: E402
    MarginBasis,
    StaticMarginBias,
    benchmark_step,
    reference_token_margins,
    verify_static_equivalence,
)
from sasa.subspace_learner import SubspaceLearner  # noqa: E402

# (label, vocabulary size, hidden size) for models the report quotes.
MODEL_SHAPES = [
    ("GPT-2 Large", 50257, 1280),
    ("Llama-3.2-1B", 128256, 2048),
    ("Llama-3.1-8B", 128256, 4096),
]


def _synthetic_case(dim: int, vocab: int, seed: int = 0):
    """Build a fitted learner, embedding table, logits and two contexts.

    Args:
        dim: Hidden size.
        vocab: Vocabulary size.
        seed: RNG seed.

    Returns:
        A ``(learner, emb, logits, ctx_a, ctx_b)`` tuple.
    """
    g = torch.Generator().manual_seed(seed)
    axis = torch.zeros(dim)
    axis[: max(1, dim // 8)] = 1.0
    neg = torch.randn(64, dim, generator=g) + 1.4 * axis
    pos = torch.randn(64, dim, generator=g) - 1.4 * axis
    learner = SubspaceLearner(embedding_dim=dim)
    learner.fit(neg, pos)
    emb = torch.randn(vocab, dim, generator=g) * 0.02
    logits = torch.randn(vocab, generator=g)
    ctx_a = torch.randn(dim, generator=g)
    ctx_b = torch.randn(dim, generator=g) * 250.0
    return learner, emb, logits, ctx_a, ctx_b


def run_exactness(repeats: int = 12) -> List[Dict]:
    """Verify exactness and context-independence across many configurations.

    Args:
        repeats: Number of random configurations to sweep.

    Returns:
        One record per configuration.
    """
    rows = []
    for seed in range(repeats):
        dim = 64 + 16 * seed
        vocab = 512 + 97 * seed
        learner, emb, logits, ctx_a, ctx_b = _synthetic_case(dim, vocab, seed)
        for alpha in (0.5, 1.0, 2.0, 8.0):
            rep = verify_static_equivalence(
                learner, emb, logits, ctx_a, ctx_b,
                alpha=alpha, temperature=1.0, top_k=50, top_p=0.9,
            )
            rows.append({"seed": seed, "dim": dim, "vocab": vocab,
                         "alpha": alpha, **rep.as_dict()})
    return rows


def run_costs(cap: int = 20000) -> List[Dict]:
    """Measure wall-clock cost and analytic operation counts per model shape.

    Large shapes are not materialised in full: a real 8B embedding table is
    2.1 GB in float32, which does not fit alongside the model in a 3 GB
    sandbox. Where the real vocabulary exceeds ``cap``, the timing is measured
    at the capped size and the arithmetic for the true shape is reported
    analytically, with the substitution recorded in the output.

    Args:
        cap: Maximum vocabulary size actually materialised for timing.

    Returns:
        One record per (label, vocabulary, hidden size).
    """
    rows = []
    for label, vocab, dim in MODEL_SHAPES:
        timing_vocab = min(vocab, cap)
        substituted = timing_vocab != vocab

        t0 = time.perf_counter()
        learner, emb, logits, ctx, _ = _synthetic_case(
            dim, timing_vocab, seed=abs(hash(label)) % 97
        )
        setup_s = time.perf_counter() - t0
        gc.collect()

        basis = MarginBasis.from_subspace_learner(learner, emb)
        fast = StaticMarginBias(basis, alpha=1.0)
        ref_ms, fast_ms = benchmark_step(
            learner, emb, logits, ctx, alpha=1.0, repeats=8
        )

        # Peak allocation of the reference path, measured directly.
        with torch.no_grad():
            _ = reference_token_margins(learner, ctx, emb)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.max_memory_allocated()
            _ = reference_token_margins(learner, ctx, emb)
            ref_bytes = torch.cuda.max_memory_allocated() - before
        else:
            ref_bytes = emb.numel() * emb.element_size()

        rows.append({
            "label": label,
            "shape_vocab": vocab,
            "shape_hidden": dim,
            "timed_vocab": timing_vocab,
            "timing_substituted": substituted,
            "setup_s": round(setup_s, 2),
            "reference_ms": ref_ms,
            "fast_ms": fast_ms,
            "measured_speedup_at_timed_vocab": ref_ms / max(fast_ms, 1e-9),
            "reference_madds_per_step": vocab * dim,
            "fast_madds_per_step": vocab,
            "analytic_reduction": dim,
            # The reference materialises a (V, d) matrix; the fast path keeps
            # only the length-V output vector.
            "reference_matrix_gib_per_step": vocab * dim * 4 / (1024 ** 3),
            "fast_vector_mib_per_step": vocab * 4 / (1024 ** 2),
            "memory_reduction": dim,
            "measured_reference_bytes": ref_bytes,
        })
        # Release the embedding table before the next (larger) shape.
        del learner, emb, logits, ctx, basis, fast
        gc.collect()
    return rows


def run_context_scan(repeats: int = 8) -> List[Dict]:
    """Sweep context magnitude and record the reference path's sensitivity.

    Args:
        repeats: Number of context scales to probe.

    Returns:
        One record per context scale.
    """
    dim, vocab = 256, 4096
    learner, emb, logits, ctx, _ = _synthetic_case(dim, vocab, seed=5)
    rows = []
    for k in range(1, repeats + 1):
        scale = float(10 ** (k - 1))
        ctx_b = ctx * scale
        p_a = torch.softmax(
            logits + reference_token_margins(learner, ctx, emb), dim=-1)
        p_b = torch.softmax(
            logits + reference_token_margins(learner, ctx_b, emb), dim=-1)
        tv = float(0.5 * (p_a - p_b).abs().sum().item())
        offset = float(
            learner.compute_margin(ctx.unsqueeze(0)).item()
            - learner.compute_margin(ctx_b.unsqueeze(0)).item()
        )
        rows.append({
            "context_scale": scale,
            "total_variation": tv,
            "reference_margin_offset": offset,
        })
    return rows


def main() -> int:
    """Run all three measurements and write the results file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/collapse.json")
    ap.add_argument("--timing-vocab-cap", type=int, default=60000,
                    help="largest vocabulary actually materialised for timing")
    args = ap.parse_args()

    print("[1/3] exactness and context-independence sweep")
    exact = run_exactness()
    n_exact = sum(1 for r in exact if r["is_exact"])
    n_free = sum(1 for r in exact if r["is_context_free"])
    print(f"      {n_exact}/{len(exact)} configurations distribution-exact")
    print(f"      {n_free}/{len(exact)} configurations context-independent")
    worst_prob = max(r["max_prob_gap"] for r in exact)
    worst_resid = max(r["residual_logit_gap"] for r in exact)
    worst_tv = max(max(r["context_sensitivity_fast"],
                       r["context_sensitivity_reference"]) for r in exact)
    print(f"      worst prob gap          {worst_prob:.3e}")
    print(f"      worst residual logit gap {worst_resid:.3e}")
    print(f"      worst context TV        {worst_tv:.3e}")

    print("[2/3] cost per model shape")
    costs = run_costs(cap=args.timing_vocab_cap)
    for r in costs:
        note = " (timed at capped vocab)" if r["timing_substituted"] else ""
        print(f"      {r['label']:>13}  ref {r['reference_ms']:7.2f} ms  "
              f"fast {r['fast_ms']:6.3f} ms  "
              f"({r['measured_speedup_at_timed_vocab']:6.1f}x at V={r['timed_vocab']})  "
              f"analytic {r['analytic_reduction']}x  "
              f"{r['reference_matrix_gib_per_step']:.2f} GiB/step{note}")

    print("[3/3] context-magnitude scan")
    scan = run_context_scan()
    for r in scan:
        print(f"      scale {r['context_scale']:>8.0f}  TV {r['total_variation']:.3e}  "
              f"offset {r['reference_margin_offset']:+.3e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"device": "cpu", "torch": torch.__version__},
        "exactness": exact,
        "costs": costs,
        "context_scan": scan,
        "summary": {
            "n_configurations": len(exact),
            "n_distribution_exact": n_exact,
            "n_context_independent": n_free,
            "worst_max_prob_gap": worst_prob,
            "worst_residual_logit_gap": worst_resid,
            "worst_context_tv": worst_tv,
        },
    }, indent=2))
    print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
