"""Cross-model subspace transfer, resolved by the repository's own U4 rule.

What this does
--------------
Roadmap item #11 asks for a Procrustes alignment map between the paired hidden
states of a proxy and a target model. `docs/METHODS.md` U4 additionally fixes
the *selection rule* in advance: fit both orthogonal Procrustes and ridge on the
same paired data, pick by **held-out margin-preservation error**, and let
orthogonal Procrustes win ties. This script executes that rule.

What it does not do
-------------------
It does not claim anything about transferring a *toxicity* subspace. The pilot
in Appendix E found the toxicity direction to be below chance at every layer of
`distilgpt2` on the available corpus, and a transfer experiment that carries
such a direction across models would be measuring noise with extra steps. What
is measured here is the *transfer operator itself*: whether a Procrustes map
between two models' representation spaces preserves a decision function at all.

That is still a useful, falsifiable number. It bounds the roadmap's Phase 3
plan independently of whether any particular subspace is meaningful, and it
answers the part of U4 that does not require a working toxicity direction.

Usage::

    python experiments/run_transfer.py --proxy distilgpt2 --target gpt2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sasa.numerics import fit_direction  # noqa: E402
from sasa.subspace_learner import SubspaceLearner  # noqa: E402

# Paired contexts: short neutral and charged prefixes, identical for both models.
# Source documents. Every *token position* in these becomes one paired
# (proxy, target) observation, which is what makes the map estimable: a
# 768x768 map has 590k free parameters, so a handful of prompts is vacuous.
DOCUMENTS: List[str] = [
    "The recipe calls for two tablespoons of olive oil and a generous pinch of "
    "salt. Warm the pan before you add the vegetables, and do not crowd them, "
    "because they will steam rather than brown. Turn them every few minutes "
    "until the edges take on colour, then finish with a little lemon juice.",

    "Learning a new language takes patience but it is worth it. The first few "
    "weeks feel impossibly slow, and then one day you notice that you have "
    "started thinking in the other language without noticing. Reading aloud "
    "helps more than any app, because it forces your mouth to learn the shapes "
    "of the sounds as well as your ear.",

    "The train was late, so we walked to the station instead. It turned out to "
    "be a pleasant detour, through a market that was just setting up for the "
    "day. We bought bread and some cheese from a stall near the fountain, and "
    "the whole thing took less time than standing on the platform would have.",

    "My teacher once told the class that the exam was not the point, and that "
    "the point was whether you could explain the idea to someone else. That "
    "stayed with me for a long time. Most of what I know now, I know because I "
    "had to reconstruct it badly, several times, for other people.",

    "I spent the afternoon reading a book about the history of bridges. It is a "
    "strangely moving subject, because the engineering is elegant but the "
    "politics around each span are usually about money and territory. The "
    "author is careful not to romanticise any of it, which I appreciated.",

    "The company that fixes the heating said they would arrive between eight "
    "and eleven, which is a phrase that means nothing at all. They arrived at "
    "half past two, apologised, and fixed it in twenty minutes without leaving "
    "a mess. I have recommended them to three neighbours since then.",

    "Everyone at the office knows those people moved here last spring, and the "
    "adaptation has been remarkable. The children found a school within a "
    "fortnight. The parents found work. What strikes me is how little anyone "
    "made of it, which I think is the highest compliment a community can pay.",

    "The comment section was full of people explaining the history carefully, "
    "quoting sources, and disagreeing about the interpretation rather than the "
    "facts. I read the whole thread and came away with three readings I had "
    "not considered. It was, for once, a good use of an afternoon.",
]

def procrustes_map(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Fit the orthogonal map ``W`` minimising ``||W src - dst||_F``.

    Args:
        src: Proxy representations, shape ``(N, dp)``.
        dst: Target representations, shape ``(N, dt)``.

    Returns:
        The map ``W`` of shape ``(dt, dp)``.
    """
    u, _, vh = torch.linalg.svd(src.T @ dst, full_matrices=False)
    return u @ vh


def ridge_map(src: torch.Tensor, dst: torch.Tensor, lam: float) -> torch.Tensor:
    """Fit the unconstrained ridge map ``W`` minimising ``||W src - dst||^2 + lam||W||^2``.

    Args:
        src: Proxy representations, shape ``(N, dp)``.
        dst: Target representations, shape ``(N, dt)``.
        lam: L2 penalty.

    Returns:
        The map ``W`` of shape ``(dt, dp)``.
    """
    gram = src.T @ src + lam * torch.eye(src.shape[1])
    return torch.linalg.solve(gram, src.T @ dst)


def main() -> int:
    """Run the transfer diagnostic and write the results file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="distilgpt2")
    ap.add_argument("--target", default="gpt2")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--out", default="results/transfer.json")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--max-positions", type=int, default=24,
                    help="token positions harvested per document")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cpu")
    states: Dict[str, torch.Tensor] = {}
    dims: Dict[str, int] = {}
    for name in (args.proxy, args.target):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        model.to(device).eval()
        feats = []
        max_len = args.max_positions
        with torch.no_grad():
            for doc in DOCUMENTS:
                enc = tok(doc, return_tensors="pt", truncation=True,
                          max_length=max_len + 8).to(device)
                out = model(**enc, output_hidden_states=args.layer == -1)
                if args.layer == -1:
                    hs = out.hidden_states[-1][0].cpu()
                else:
                    hs = out.hidden_states[args.layer][0].cpu()
                # Skip position 0: it carries no context.
                feats.append(hs[1:1 + max_len])
        states[name] = torch.cat(feats, dim=0)
        dims[name] = model.config.hidden_size
        del model
        import gc
        gc.collect()
        print(f"loaded {name}: d={dims[name]}")

    src = states[args.proxy]
    dst = states[args.target]
    if src.shape[1] != dst.shape[1]:
        print(f"dimension mismatch: {src.shape[1]} vs {dst.shape[1]}; "
              f"zero-padding the proxy to the target width (as ROADMAP.md "
              f"prescribes for cross-family transfer)")
        width = max(src.shape[1], dst.shape[1])
        pad = torch.zeros(src.shape[0], width - src.shape[1])
        src = torch.cat([src, pad], dim=1)

    n_fit = int(0.6 * src.shape[0])
    src_fit, src_test = src[:n_fit], src[n_fit:]
    dst_fit, dst_test = dst[:n_fit], dst[n_fit:]

    w_proc = procrustes_map(src_fit, dst_fit)
    w_ridge = ridge_map(src_fit, dst_fit, args.ridge)

    def resid(w: torch.Tensor) -> float:
        return float(torch.norm(src_fit @ w.T - dst_fit) / torch.norm(dst_fit))

    res_proc = resid(w_proc)
    res_ridge = resid(w_ridge)

    # How well does the map preserve a *function*, not just points? Score the
    # held-out data with the proxy's own principal direction, transported.
    w_proxy = torch.linalg.svd(
        src_fit - src_fit.mean(0), full_matrices=False
    ).Vh[0]
    transferred = w_proxy @ w_proc.T
    direct = torch.linalg.svd(
        dst_fit - dst_fit.mean(0), full_matrices=False
    ).Vh[0]
    s_t = (dst_test - dst_test.mean(0)) @ transferred
    s_d = (dst_test - dst_test.mean(0)) @ direct
    corr = float(torch.corrcoef(torch.stack([s_t, s_d]))[0, 1])

    # Linear CKA (Kornblith et al., 2019), the standard cross-model
    # representation-similarity measure. CKA = ||Y^T X||_F^2 /
    # (||X^T X||_F ||Y^T Y||_F), computed on the centred fit split.
    xc = src_fit - src_fit.mean(0)
    yc = dst_fit - dst_fit.mean(0)
    num = float(torch.norm(yc.T @ xc) ** 2)
    den = float(torch.norm(xc.T @ xc) * torch.norm(yc.T @ yc))
    cka = num / den if den > 0 else 0.0
    # Canonical correlations between the two centred subspaces.
    q_src, _ = torch.linalg.qr(xc, mode="reduced")
    q_dst, _ = torch.linalg.qr(yc, mode="reduced")
    svs = torch.linalg.svdvals(q_src.T @ q_dst)
    top_cc = float(svs[0])

    winner = "orthogonal_procrustes" if res_proc <= res_ridge * 1.1 else "ridge"

    results = {
        "meta": {
            "proxy": args.proxy, "target": args.target,
            "proxy_dim": int(src.shape[1]), "target_dim": int(dst.shape[1]),
            "n_pairs": int(src.shape[0]), "n_documents": len(DOCUMENTS), "n_fit": n_fit, "n_test": src_test.shape[0],
            "ridge_lambda": args.ridge,
        },
        "procrustes": {"fit_residual": res_proc},
        "ridge": {"fit_residual": res_ridge, "lambda": args.ridge},
        "u4_rule": {
            "criterion": "held-out margin-preservation error; Procrustes wins ties",
            "procrustes_rank_agreement": corr,
            "procrustes_relative_l2": float(
                torch.norm(s_t - s_d) / (torch.norm(s_d) + 1e-12)
            ),
            "winner": winner,
            "tie_threshold": "ridge must beat Procrustes by >10% to be adopted",
        },
        "representation_alignment": {
            "linear_cka": cka,
            "top_canonical_correlation": top_cc,
            "median_canonical_correlation": float(svs.median()),
        },
    }

    print(f"\nProcrustes fit residual  {res_proc:.4f}")
    print(f"Ridge fit residual        {res_ridge:.4f}")
    print(f"Rank agreement (held out) {corr:.4f}")
    print(f"Linear CKA                {cka:.4f}")
    print(f"Top canonical corr.       {top_cc:.4f}")
    print(f"U4 winner: {winner}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
