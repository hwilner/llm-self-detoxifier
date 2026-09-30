"""Experiment: what the decode loop actually costs, cached and batched.

Two ideas that are both about the loop rather than the subspace, and that the
repository's efficiency claims cannot be stated without.

**Idea 15 -- the key-value cache.** ``SASASampler.generate`` re-runs a full
forward pass over the whole prefix at every step, so generating ``L`` tokens
costs ``O(L^2)`` forward work. Appendix B removes the per-token margin cost
exactly; this measures the *other* term, which is the one that dominates.

**Idea 18 -- batching.** The steering vector is context-free, so one ``(V,)``
add serves every row of a batch. A context-dependent margin would need one
contraction per row, which is the main reason batching a steered decoder is
otherwise awkward.

Both are verified for *exactness* before being timed: the cached loop must
reproduce the uncached token sequence, and a batch must reproduce independent
single-row draws. A speed-up measured against an implementation that computes
something different is not a speed-up.

Usage::

    .venv/bin/python experiments/run_decode_costs.py --out results/decode_costs.json
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

from sasa.decoding import (  # noqa: E402
    batched_sample,
    cached_generate,
    token_budget,
    uncached_generate,
)
from sasa.fast_margin import MarginBasis  # noqa: E402

PROMPTS: List[str] = [
    "I think that people who", "Honestly, anyone who",
    "The problem with that group is that they",
    "My neighbour keeps complaining and", "She told me that",
    "The whole situation made me feel like", "What I hate about this city is",
    "They always say that immigrants", "The recipe calls for two tablespoons of",
    "Learning a new language takes patience", "The train was late, so we walked to",
    "My teacher once told the class that",
]


def _time(fn, repeats: int) -> float:
    """Return the mean wall-clock seconds per call.

    Args:
        fn: Zero-argument callable to time.
        repeats: Number of timed iterations.

    Returns:
        Mean seconds per call.
    """
    fn()  # warm up
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def run_caching(model, tokenizer, device, bias, prompt_lens, gen_lens) -> Dict:
    """Time cached against uncached decoding and check they agree exactly.

    Args:
        model: A causal language model.
        tokenizer: The matching tokenizer.
        device: Device to run on.
        bias: Static steering vector, or ``None``.
        prompt_lens: Prompt lengths to sweep.
        gen_lens: Generation lengths to sweep.

    Returns:
        A dict of per-configuration timings, exactness checks, and the
        analytic token-position counts.
    """
    rows = []
    for p_len, g_len in zip(prompt_lens, gen_lens):
        prompt = " ".join(["word"] * max(1, p_len))
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        gen = torch.Generator().manual_seed(7)
        a = uncached_generate(model, ids, g_len, bias=bias, top_k=50, generator=gen)
        gen = torch.Generator().manual_seed(7)
        b = cached_generate(model, ids, g_len, bias=bias, top_k=50, generator=gen)

        repeats = 1 if g_len > 48 else 3
        ref_s = _time(
            lambda: uncached_generate(model, ids, g_len, bias=bias, top_k=50,
                                      generator=torch.Generator().manual_seed(7)),
            repeats)
        cch_s = _time(
            lambda: cached_generate(model, ids, g_len, bias=bias, top_k=50,
                                    generator=torch.Generator().manual_seed(7)),
            repeats)
        budget = token_budget(int(ids.shape[1]), g_len, n_layers=6)
        rows.append({
            "prompt_len": int(ids.shape[1]),
            "gen_len": g_len,
            "uncached_ms": 1000 * ref_s,
            "cached_ms": 1000 * cch_s,
            "measured_speedup": ref_s / max(cch_s, 1e-9),
            "tokens_identical": bool(torch.equal(a.tokens, b.tokens)),
            "logit_norms_max_abs_diff": float(
                (a.step_logits_norm - b.step_logits_norm).abs().max().item()
            ),
            **budget,
        })
        print(f"      prompt={rows[-1]['prompt_len']:3d} gen={g_len:3d}  "
              f"uncached {rows[-1]['uncached_ms']:8.1f} ms  "
              f"cached {rows[-1]['cached_ms']:7.1f} ms  "
              f"{rows[-1]['measured_speedup']:6.2f}x  "
              f"identical={rows[-1]['tokens_identical']}")
    return {
        "rows": rows,
        "all_tokens_identical": all(r["tokens_identical"] for r in rows),
        "max_logit_norm_diff": max(r["logit_norms_max_abs_diff"] for r in rows),
        "median_measured_speedup": sorted(
            r["measured_speedup"] for r in rows
        )[len(rows) // 2],
        "analytic_speedup_at_longest": rows[-1]["speedup"],
    }


def run_batching(base_logits: torch.Tensor, bias, batch_sizes) -> Dict:
    """Check that batched sampling equals independent single-row draws.

    Args:
        base_logits: A pool of per-prompt logits, shape ``(P, V)``.
        bias: Static steering vector, or ``None``.
        batch_sizes: Batch sizes to test.

    Returns:
        A dict of per-size agreement and timing.
    """
    rows = []
    for b in batch_sizes:
        logits = base_logits[:b]
        # Exactness: the per-row-generator path must reproduce independent
        # single-row draws.
        gens = [torch.Generator().manual_seed(1000 + i) for i in range(b)]
        batched = batched_sample(logits, bias=bias, generators=gens)
        singles = [
            int(batched_sample(logits[i:i + 1], bias=bias,
                               generators=[torch.Generator().manual_seed(1000 + i)])[0])
            for i in range(b)
        ]
        exact = batched.tolist() == singles

        # Throughput: the vectorised path, which is the whole point of idea 18.
        reps = 300
        t0 = time.perf_counter()
        for _ in range(reps):
            batched_sample(logits, bias=bias,
                           generator=torch.Generator().manual_seed(7))
        vec = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            for i in range(b):
                batched_sample(logits[i:i + 1], bias=bias,
                               generator=torch.Generator().manual_seed(7))
        loop = (time.perf_counter() - t0) / reps
        rows.append({
            "batch_size": b,
            "matches_independent_rows": exact,
            "vectorised_ms": vec * 1000,
            "per_row_loop_ms": loop * 1000,
            "vectorised_ms_per_item": vec * 1000 / b,
            "per_row_loop_ms_per_item": loop * 1000 / b,
            "vectorised_speedup_vs_loop": loop / max(vec, 1e-12),
        })
        print(f"      batch={b:3d}  exact={exact}  "
              f"vectorised {vec * 1000:6.2f} ms ({vec * 1000 / b:5.3f}/item)  "
              f"loop {loop * 1000:7.2f} ms ({loop * 1000 / b:5.3f}/item)  "
              f"{loop / max(vec, 1e-12):5.2f}x")
    return {
        "rows": rows,
        "all_match": all(r["matches_independent_rows"] for r in rows),
        "max_vectorised_speedup": max(
            r["vectorised_speedup_vs_loop"] for r in rows),
    }


def main() -> int:
    """Run both studies and write the results file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--out", default="results/decode_costs.json")
    ap.add_argument("--subspace", default="results/subspace.pt")
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device).eval()
    emb = model.get_input_embeddings().weight.detach()
    d = model.config.hidden_size

    bias = None
    if Path(args.subspace).exists():
        state = torch.load(args.subspace, map_location="cpu", weights_only=True)
        basis = MarginBasis(
            weight=state["weight"], bias=state["bias"],
            token_weights=emb @ state["weight"],
            embedding_dim=d, vocab_size=emb.shape[0],
        )
        bias = basis.static_bias(8.0)
        print(f"steering at alpha=8: mean |shift| = {float(bias.abs().mean()):.4f}")
    else:
        print("no subspace found; timing the unsteered loop only")

    print("[1/2] caching: exactness then timing")
    caching = run_caching(
        model, tokenizer, device, bias,
        prompt_lens=[4, 8, 16], gen_lens=[8, 16, 32],
    )

    print("[2/2] batching: exactness then timing")
    with torch.no_grad():
        rows = []
        for text in PROMPTS:
            enc = tokenizer(text, return_tensors="pt").to(device)
            rows.append(model(**enc).logits[0, -1, :].cpu())
    base_logits = torch.stack(rows)
    sizes = [b for b in (1, 2, 4, 8, 16, 32) if b <= base_logits.shape[0]]
    batching = run_batching(base_logits, bias, batch_sizes=sizes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"model": args.model, "hidden": d, "vocab": int(emb.shape[0]),
                 "steered": bias is not None, "seed": args.seed},
        "caching": caching,
        "batching": batching,
    }, indent=2))
    gc.collect()
    print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
