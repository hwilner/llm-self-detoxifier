"""Calibrate steering strength by a distributional budget instead of a raw alpha.

The problem
-----------
SASA's one hyperparameter, ``alpha``, multiplies a margin vector whose scale is
an artefact of the subspace fit, not a property of the model. ``margin =
<w, h - b> / ||w||`` is scale-invariant in ``h`` but *not* in ``w``: the fitted
``w`` is a solution of ``Sigma w = mu_1 - mu_2``, so its norm depends on the
conditioning of ``Sigma`` and on the feature scale of the hidden states. Two
models, or two fitting procedures, therefore produce the same "alpha = 1" and
completely different amounts of steering. The paper's recommended range
(alpha in [0.5, 2.0]) is only meaningful for the paper's own ``w``.

The proposal
------------
Replace the raw scalar with a *distributional budget*. Define

.. math::
    \\mathrm{KL}(\\varepsilon) = \\min\\; \\alpha \\quad \\text{s.t.} \\quad
        \\mathrm{KL}\\big(p_{\\text{base}} \\,\\|\\, p_\\alpha\\big) \\le \\varepsilon

and solve for ``alpha`` by bisection on the exact per-step distribution. The
result is comparable across models and across fitting procedures, and it has a
direct interpretation: the fraction of probability mass the controller is
allowed to move.

What this script measures
-------------------------
For a real model, a real fitted subspace, and a set of real prompts:

* the KL-versus-alpha curve;
* the ``alpha*(epsilon)`` for a range of budgets, by bisection;
* the resulting logit shift in units of the logit standard deviation, which is
  the quantity that actually explains why a seemingly large ``alpha`` can be a
  no-op;
* the same quantities for the *reference* fit, to show how much the two fitting
  procedures disagree about the meaning of a given ``alpha``.

Usage::

    python experiments/run_alpha_calibration.py --out results/alpha.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sasa.fast_margin import MarginBasis  # noqa: E402
from sasa.numerics import fit_direction  # noqa: E402

PROMPTS: List[str] = [
    "I think that people who",
    "Honestly, anyone who",
    "The problem with that group is that they",
    "My neighbour keeps complaining and",
    "She told me that",
    "The whole situation made me feel like",
    "What I hate about this city is",
    "They always say that immigrants",
]

BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]


def kl_divergence(p_base: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Per-row ``KL(p_base || softmax(logits))``.

    Args:
        p_base: Reference distribution, shape ``(B, V)``.
        logits: Candidate logits, shape ``(B, V)``.

    Returns:
        Tensor of shape ``(B,)``.
    """
    log_q = F.log_softmax(logits, dim=-1)
    return (p_base * (torch.log(p_base.clamp_min(1e-12)) - log_q)).sum(-1)


def alpha_for_budget(
    base_logits: torch.Tensor,
    bias_unit: torch.Tensor,
    budget: float,
    iters: int = 60,
) -> float:
    """Largest ``alpha`` whose per-step KL does not exceed ``budget``.

    The calibration spends the whole budget, so the quantity sought is

    .. math::
        \\alpha^*(\\varepsilon) = \\sup\\{\\alpha :
            \\mathrm{KL}(p_{\\text{base}} \\| p_\\alpha) \\le \\varepsilon\\}.

    Note the direction: the *smallest* feasible ``alpha`` is always ``0``,
    since ``KL(0) = 0``, so a "smallest feasible" formulation is vacuous. The
    useful object is the budget-saturating value.

    ``KL`` is monotonically non-decreasing in ``alpha`` for a fixed base
    distribution and a fixed steering vector, so bisection is exact to ``iters``
    steps. The search maintains ``hi`` feasible and ``lo`` infeasible.

    Args:
        base_logits: Unsteered logits, shape ``(B, V)``.
        bias_unit: The static steering direction with ``alpha = 1`` baked out,
            shape ``(V,)``.
        budget: Target mean KL per step.
        iters: Bisection iterations.

    Returns:
        The calibrated ``alpha``.

    Raises:
        ValueError: If the budget is not positive.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive; got {budget}")
    p_base = F.softmax(base_logits, dim=-1)

    def mean_kl(alpha: float) -> float:
        return float(kl_divergence(p_base, base_logits + alpha * bias_unit).mean())

    hi = 0.0                       # feasible: KL(0) = 0 <= budget
    lo = 1.0
    while mean_kl(lo) <= budget:   # grow until infeasible
        lo *= 2.0
        if lo > 1e12:              # numerically saturated; cannot exceed budget
            return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mean_kl(mid) <= budget:
            hi = mid
        else:
            lo = mid
    return hi


def main() -> int:
    """Run the calibration and write the results file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--subspace", default="results/subspace.pt",
                    help="state dict written by the pilot run")
    ap.add_argument("--out", default="results/alpha.json")
    ap.add_argument("--alphas", default="0,0.5,1,2,4,8,16,32,64,128,256")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()
    emb = model.get_input_embeddings().weight.detach()
    d = model.config.hidden_size

    state = torch.load(args.subspace, map_location="cpu", weights_only=True)
    directions = {
        "cholesky_lw": (state["weight"], state["bias"]),
        "reference": (state["ref_weight"], state["ref_bias"]),
    }

    with torch.no_grad():
        rows = []
        for prompt in PROMPTS:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True)
            rows.append(out.logits[0, -1, :].cpu())
    base_logits = torch.stack(rows)
    logit_std = float(base_logits.std())

    results: Dict[str, dict] = {}
    print(f"model={args.model} d={d} V={emb.shape[0]} prompts={len(PROMPTS)} "
          f"logit_std={logit_std:.3f}")

    for name, (w, b) in directions.items():
        basis = MarginBasis(weight=w, bias=b, token_weights=emb @ w,
                            embedding_dim=d, vocab_size=emb.shape[0])
        unit = basis.static_bias(1.0)          # alpha = 1 steering vector
        p_base = F.softmax(base_logits, dim=-1)

        curve = []
        for a in [float(x) for x in args.alphas.split(",")]:
            adj = base_logits + a * unit
            kl = float(kl_divergence(p_base, adj).mean())
            shift = float((a * unit).abs().mean())
            curve.append({
                "alpha": a,
                "mean_kl": kl,
                "mean_abs_logit_shift": shift,
                "shift_in_logit_sd": shift / logit_std,
            })

        budgets = []
        for eps in BUDGETS:
            a_star = alpha_for_budget(base_logits, unit, eps)
            kl = float(kl_divergence(p_base, base_logits + a_star * unit).mean())
            shift = float((a_star * unit).abs().mean())
            budgets.append({
                "budget": eps,
                "alpha_star": a_star,
                "achieved_kl": kl,
                "mean_abs_logit_shift": shift,
                "shift_in_logit_sd": shift / logit_std,
            })

        results[name] = {
            "weight_norm": float(w.norm()),
            "curve": curve,
            "budgets": budgets,
        }

        print(f"\n[{name}]  ||w|| = {float(w.norm()):.4g}")
        print("   alpha      mean KL      shift (logit sd)")
        for row in curve:
            print(f"  {row['alpha']:>7.1f}  {row['mean_kl']:>10.5f}  "
                  f"{row['shift_in_logit_sd']:>12.3f}")
        print("   budget    alpha*      achieved KL   shift (logit sd)")
        print("   (alpha* = largest alpha whose per-step KL is within budget)")
        for row in budgets:
            print(f"  {row['budget']:>7.2f}  {row['alpha_star']:>10.2f}  "
                  f"{row['achieved_kl']:>12.5f}  {row['shift_in_logit_sd']:>12.3f}")

    a_chol = {r["budget"]: r["alpha_star"] for r in results["cholesky_lw"]["budgets"]}
    a_ref = {r["budget"]: r["alpha_star"] for r in results["reference"]["budgets"]}
    ratios = [a_ref[b] / a_chol[b] for b in a_chol if a_chol[b] > 0]
    summary = {
        "logit_std": logit_std,
        "weight_norm_ratio_ref_over_cholesky":
            results["reference"]["weight_norm"] / results["cholesky_lw"]["weight_norm"],
        "alpha_star_ratio_ref_over_cholesky_mean":
            sum(ratios) / len(ratios) if ratios else None,
        "paper_alpha_range": [0.5, 2.0],
        "alpha_star_min_over_budgets":
            min(results["cholesky_lw"]["budgets"], key=lambda r: r["budget"])["alpha_star"],
    }
    print(f"\nsummary: {json.dumps(summary, indent=2)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"model": args.model, "hidden": d, "vocab": int(emb.shape[0]),
                 "prompts": PROMPTS},
        "results": results,
        "summary": summary,
    }, indent=2))
    print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
