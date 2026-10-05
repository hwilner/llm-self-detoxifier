#!/usr/bin/env python3
"""
Cross-architecture linear-separability probe (issue #40).

Question: is toxicity as linearly separable in SSM/hybrid recurrent states
(Mamba, Jamba) as it is in Transformer activations? This determines whether
SASA's Fisher/LDA subspace — and Phase-3 cross-paradigm transfer — is viable
beyond Transformers. The same probe doubles as the tamper-detection signal
for #47 (separability collapse after fine-tuning/abliteration = tampering).

Usage (real models, requires transformers + a labeled dataset):

    python experiments/separability_probe.py \
        --models gpt2 state-spaces/mamba-130m-hf \
        --non-toxic data/non_toxic.txt --toxic data/toxic.txt \
        --layers all --val-fraction 0.25

Usage (offline pipeline validation, no downloads):

    python experiments/separability_probe.py --synthetic

Output: a per-model, per-layer held-out separability table (accuracy of the
closed-form Fisher/LDA classifier on a validation split), printed and written
to --out (JSON).
"""

import argparse
import json
import sys
from typing import Dict, List

import torch


def separability_table(
    train_nt: Dict[int, torch.Tensor],
    train_t: Dict[int, torch.Tensor],
    val_nt: Dict[int, torch.Tensor],
    val_t: Dict[int, torch.Tensor],
) -> Dict[int, float]:
    """Fit per-layer Fisher/LDA subspaces and score held-out accuracy."""
    # Local import so --synthetic works without the full package installed.
    from sasa.multilayer import MultiLayerSubspaceLearner

    layers = sorted(train_nt.keys())
    dim = train_nt[layers[0]].shape[1]
    learner = MultiLayerSubspaceLearner(dim, layers, selection="best")
    learner.fit(train_nt, train_t)
    return learner.separability_scores(val_nt, val_t)


def split(
    embeddings: Dict[int, torch.Tensor], val_fraction: float, seed: int = 0
):
    """Deterministic per-layer train/val split (same indices every layer)."""
    n = next(iter(embeddings.values())).shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train = {l: e[train_idx] for l, e in embeddings.items()}
    val = {l: e[val_idx] for l, e in embeddings.items()}
    return train, val


def run_synthetic() -> Dict[str, Dict[int, float]]:
    """
    Validate the probe pipeline without model downloads.

    Simulates two architectures: 'transformer-like' (toxic shift concentrated
    in a few mid-layer directions -> highly separable) and 'ssm-like' (shift
    spread diffusely across all directions -> less separable by a linear probe
    of the same norm). This checks the plumbing end-to-end; it makes NO claim
    about real models.
    """
    torch.manual_seed(0)
    dim, n, layers = 64, 256, [0, 1, 2, 3]

    def make_model(shift_vec_by_layer):
        nt, t = {}, {}
        for l in layers:
            shift = shift_vec_by_layer[l]
            nt[l] = torch.randn(n, dim)
            t[l] = torch.randn(n, dim) + shift
        return nt, t

    concentrated = {l: (torch.tensor([3.0] + [0.0] * (dim - 1)) if l == 2 else torch.zeros(dim)) for l in layers}
    diffuse = {l: torch.full((dim,), 3.0 / dim ** 0.5) * (1.0 if l == 2 else 0.0) for l in layers}

    results = {}
    for name, shifts in [("synthetic-transformer-like", concentrated), ("synthetic-ssm-like", diffuse)]:
        nt, t = make_model(shifts)
        tr_nt, v_nt = split(nt, 0.25)
        tr_t, v_t = split(t, 0.25)
        results[name] = separability_table(tr_nt, tr_t, v_nt, v_t)
    return results


def run_models(args) -> Dict[str, Dict[int, float]]:
    """Real-model mode: extract per-layer embeddings and score separability."""
    from transformers import AutoModel, AutoTokenizer  # noqa: delayed import
    from sasa.multilayer import extract_embeddings_multilayer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.non_toxic) as f:
        non_toxic_texts = [l.strip() for l in f if l.strip()]
    with open(args.toxic) as f:
        toxic_texts = [l.strip() for l in f if l.strip()]

    results = {}
    for model_name in args.models:
        print(f"[probe] loading {model_name} ...", file=sys.stderr)
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]

        nt = extract_embeddings_multilayer(model, tokenizer, non_toxic_texts, device, layers)
        t = extract_embeddings_multilayer(model, tokenizer, toxic_texts, device, layers)
        tr_nt, v_nt = split(nt, args.val_fraction, seed=args.seed)
        tr_t, v_t = split(t, args.val_fraction, seed=args.seed)
        results[model_name] = separability_table(tr_nt, tr_t, v_nt, v_t)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def print_table(results: Dict[str, Dict[int, float]]) -> None:
    print(f"\n{'model':40s} " + " ".join(f"layer {l:>2d}" for l in sorted(next(iter(results.values())).keys())))
    for name, scores in results.items():
        row = " ".join(f"{scores[l]:8.3f}" for l in sorted(scores.keys()))
        print(f"{name:40s} {row}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", help="HF model ids (e.g. gpt2 state-spaces/mamba-130m-hf)")
    p.add_argument("--non-toxic", help="Path to newline-separated non-toxic texts")
    p.add_argument("--toxic", help="Path to newline-separated toxic texts")
    p.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="separability_results.json")
    p.add_argument("--synthetic", action="store_true", help="Offline pipeline validation (no downloads)")
    args = p.parse_args()

    if args.synthetic:
        results = run_synthetic()
    else:
        if not (args.models and args.non_toxic and args.toxic):
            p.error("real-model mode requires --models, --non-toxic and --toxic")
        results = run_models(args)

    print_table(results)
    with open(args.out, "w") as f:
        json.dump({k: {str(l): s for l, s in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
