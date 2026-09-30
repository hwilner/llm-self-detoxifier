"""Experiment: where does subspace transfer break?

Roadmap item #11 and idea 11. Appendix F established, for one same-family pair,
that a Procrustes map is estimable from a few hundred paired states while a ridge
map is not. What that leaves open is the *boundary*: how the transfer quality
varies with the pair, and whether some cheap statistic predicts it.

If a statistic does, then the expensive part of Phase 3 -- fitting a map --
becomes a cheap screening step, in the same spirit as the static interference
prediction in :mod:`sasa.multi_attribute`. If none does, then the 2-5 k paired
states in ``docs/ROADMAP.md`` are a bet rather than a plan, and the honest
report says so.

For each ordered model pair this measures:

* the Procrustes fit residual;
* linear CKA between the two representation spaces (Kornblith et al., 2019);
* the **held-out rank agreement** of the transferred top principal direction,
  which is the margin-preservation quantity U4 actually selects on.

and then reports the rank correlation of CKA against transfer quality. The
correlate is over a handful of pairs, so the result is a direction, not a
coefficient, and it is reported that way.

Usage::

    .venv/bin/python experiments/run_transfer_boundary.py --out results/transfer_boundary.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.run_multi_attribute import spearman  # noqa: E402

#: Source documents; every token position becomes one paired observation.
DOCUMENTS: List[str] = [
    "The recipe calls for two tablespoons of olive oil and a generous pinch of "
    "salt. Warm the pan before you add the vegetables, and do not crowd them, "
    "because they will steam rather than brown. Finish with a little lemon.",

    "Learning a new language takes patience but it is worth it. The first few "
    "weeks feel impossibly slow, and then one day you notice that you have "
    "started thinking in the other language without noticing at all.",

    "The train was late, so we walked to the station instead. It turned out to "
    "be a pleasant detour, through a market that was just setting up for the "
    "day. We bought bread and some cheese near the old fountain.",

    "My teacher once told the class that the exam was not the point, and that "
    "the point was whether you could explain the idea to someone else. That "
    "stayed with me for a long time afterwards, for better and for worse.",

    "I spent the afternoon reading a book about the history of bridges. It is a "
    "strangely moving subject, because the engineering is elegant but the "
    "politics around each span are usually about money and territory.",

    "The company that fixes the heating said they would arrive between eight "
    "and eleven, which is a phrase that means nothing at all. They arrived at "
    "half past two and fixed it in twenty minutes without leaving a mess.",

    "Everyone at the office knows those people moved here last spring, and the "
    "adaptation has been remarkable. What strikes me is how little anyone made "
    "of it, which I think is the highest compliment a community can pay.",

    "The comment section was full of people explaining the history carefully, "
    "quoting sources, and disagreeing about the interpretation rather than the "
    "facts. I read the whole thread and came away with three new readings.",
]


#: Explicit model families. A prefix comparison gets this wrong: ``distilgpt2``
#: is distilled *from* ``gpt2``, so the pair is same-family even though the
#: strings share no prefix.
FAMILY = {
    "distilgpt2": "gpt2-lineage",
    "gpt2": "gpt2-lineage",
    "gpt2-medium": "gpt2-lineage",
    "gpt2-large": "gpt2-lineage",
    "gpt2-xl": "gpt2-lineage",
    "EleutherAI_pythia-410m": "pythia",
    "EleutherAI_pythia-1.4b": "pythia",
}


def _collect(model, tokenizer, device, max_positions: int) -> torch.Tensor:
    """Harvest last-layer hidden states at every token position of the documents.

    Args:
        model: A causal language model.
        tokenizer: The matching tokenizer.
        device: Device to run on.
        max_positions: Positions kept per document.

    Returns:
        Tensor of shape ``(n_documents * max_positions, d)``.
    """
    out = []
    with torch.no_grad():
        for doc in DOCUMENTS:
            enc = tokenizer(doc, return_tensors="pt", truncation=True,
                            max_length=max_positions + 8).to(device)
            res = model(**enc, output_hidden_states=True)
            # Some checkpoints (pythia among them) load in float16; cast so
            # that pairs from different dtypes can be cross-multiplied.
            hs = res.hidden_states[-1][0].float().cpu()
            out.append(hs[1:1 + max_positions])
    return torch.cat(out, dim=0)


def _zero_pad(x: torch.Tensor, width: int) -> torch.Tensor:
    """Zero-pad a feature matrix on the right to a target width.

    Args:
        x: Features of shape ``(N, d)``.
        width: Target width.

    Returns:
        Padded features of shape ``(N, width)``.

    Raises:
        ValueError: If ``width`` is smaller than the current width.
    """
    if x.shape[1] == width:
        return x
    if width < x.shape[1]:
        raise ValueError(f"cannot shrink {x.shape[1]} to {width}")
    pad = torch.zeros(x.shape[0], width - x.shape[1], dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two centred feature matrices.

    Args:
        x: Features of shape ``(N, d1)``.
        y: Features of shape ``(N, d2)``.

    Returns:
        CKA in ``[0, 1]``.
    """
    xc, yc = x - x.mean(0), y - y.mean(0)
    num = float(torch.norm(yc.T @ xc) ** 2)
    den = float(torch.norm(xc.T @ xc) * torch.norm(yc.T @ yc))
    return num / den if den > 0 else 0.0


def main() -> int:
    """Run every ordered pair and write the results file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="distilgpt2,gpt2,gpt2-medium")
    ap.add_argument("--max-positions", type=int, default=24)
    ap.add_argument("--out", default="results/transfer_boundary.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cpu")
    names = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"[1/2] harvesting {args.max_positions} positions x "
          f"{len(DOCUMENTS)} documents per model")
    states: Dict[str, torch.Tensor] = {}
    dims: Dict[str, int] = {}
    for name in names:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        model.to(device).eval()
        states[name] = _collect(model, tok, device, args.max_positions)
        dims[name] = model.config.hidden_size
        print(f"      {name:14s} d={dims[name]:5d} "
              f"states={tuple(states[name].shape)}")
        del model
        gc.collect()

    width = max(dims.values())
    padded = {k: _zero_pad(v, width) for k, v in states.items()}

    print("[2/2] ordered pairs")
    rows: List[Dict[str, object]] = []
    for proxy, target in [(p, t) for p in names for t in names if p != t]:
        src, dst = padded[proxy], padded[target]
        n_fit = int(0.6 * src.shape[0])
        src_fit, src_test = src[:n_fit], src[n_fit:]
        dst_fit, dst_test = dst[:n_fit], dst[n_fit:]

        # Orthogonal Procrustes via the SVD of the cross-covariance.
        u, _s, vh = torch.linalg.svd(src_fit.T @ dst_fit, full_matrices=False)
        w = u @ vh
        residual = float(
            torch.norm(src_fit @ w.T - dst_fit) / torch.norm(dst_fit)
        )

        # The top principal direction of each side, and its transfer.
        v_proxy = torch.linalg.svd(
            src_fit - src_fit.mean(0), full_matrices=False).Vh[0]
        transferred = v_proxy @ w.T
        direct = torch.linalg.svd(
            dst_fit - dst_fit.mean(0), full_matrices=False).Vh[0]
        s_t = (dst_test - dst_test.mean(0)) @ transferred
        s_d = (dst_test - dst_test.mean(0)) @ direct
        corr_signed = float(torch.corrcoef(torch.stack([s_t, s_d]))[0, 1])
        # A principal direction has an arbitrary sign, so a perfectly
        # transported direction can score -1. Reporting the signed value would
        # call a correct transfer a catastrophic failure, so the absolute value
        # is the meaningful one and the signed value is kept alongside it.
        corr = abs(corr_signed)
        cka = _linear_cka(src_fit, dst_fit)

        same_family = FAMILY.get(proxy) == FAMILY.get(target, proxy)
        rows.append({
            "proxy": proxy, "target": target,
            "proxy_dim": dims[proxy], "target_dim": dims[target],
            "padded_to": width,
            "zero_padding_needed": dims[proxy] != dims[target],
            "same_family": same_family,
            "n_pairs": int(src.shape[0]), "n_fit": n_fit,
            "procrustes_residual": residual,
            "heldout_rank_agreement": corr,
            "heldout_rank_agreement_signed": corr_signed,
            "linear_cka": cka,
        })
        print(f"      {proxy:13s} -> {target:13s} residual={residual:6.3f}  "
              f"agreement={corr:+.3f}  CKA={cka:.3f}  "
              f"padded={rows[-1]['zero_padding_needed']}")

    quality = [1.0 - r["heldout_rank_agreement"] for r in rows]
    ckas = [r["linear_cka"] for r in rows]
    resids = [r["procrustes_residual"] for r in rows]
    rho_quality, p_quality = spearman(ckas, quality)
    rho_resid, p_resid = spearman(ckas, resids)

    same = [1.0 - r["heldout_rank_agreement"] for r in rows if r["same_family"]]
    cross = [1.0 - r["heldout_rank_agreement"] for r in rows if not r["same_family"]]

    print(f"      Spearman(CKA, transfer error)   rho={rho_quality:+.3f} p={p_quality:.3f}")
    print(f"      Spearman(CKA, fit residual)     rho={rho_resid:+.3f} p={p_resid:.3f}")
    if same:
        print(f"      mean transfer error, same family : {sum(same) / len(same):.4f} "
              f"(n={len(same)})")
    if cross:
        print(f"      mean transfer error, cross family: {sum(cross) / len(cross):.4f} "
              f"(n={len(cross)})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"models": names, "dims": dims, "padded_to": width,
                 "max_positions": args.max_positions,
                 "n_documents": len(DOCUMENTS)},
        "pairs": rows,
        "correlation": {
            "cka_vs_transfer_error": {"rho": rho_quality, "p": p_quality},
            "cka_vs_fit_residual": {"rho": rho_resid, "p": p_resid},
        },
        "mean_transfer_error": {
            "same_family": (sum(same) / len(same)) if same else None,
            "cross_family": (sum(cross) / len(cross)) if cross else None,
        },
        "caveat": ("the transferred direction is the top principal component of "
                   "the fit split, which is dominated by frequency and position "
                   "effects; this measures the transfer operator, not whether a "
                   "semantic subspace would survive it"),
    }, indent=2))
    print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
