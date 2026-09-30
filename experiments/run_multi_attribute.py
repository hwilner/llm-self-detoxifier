"""Experiment: multi-attribute interference, graded ranks, and confounds.

Three ideas share one script because they share one corpus.

1. **Idea 6 -- static interference prediction.** For every pair of attribute
   subspaces, predict interference from geometry alone (subspace overlap), then
   *measure* interference independently by composing biases at matched steering
   mass and measuring how much of each attribute's intended margin effect
   survives. The claim is that the static prediction ranks the measured
   interference. A Spearman correlation is the test, and the null is "no
   relationship".

   Measuring retention from logits rather than from generations is deliberate.
   Appendix E showed that generation-based measurement on this model is
   underpowered, whereas retention is an exact, deterministic function of the
   fitted directions -- so a null here is a real null, not a power problem.

2. **Idea 8 -- graded supervision is free and richer.** The externally judged
   corpus carries a 0-4 grade, which the binary contrast throws away. Fitting on
   the full grade should give k-1 discriminant directions and should predict the
   *grade* better than a binary split can, at no extra labelling cost.

3. **Idea 9 -- confounds.** Fit named probes and report the angle between each
   and the others, with a significance test for whether each probe is
   identifiable at all.

Usage::

    .venv/bin/python experiments/run_multi_attribute.py --out results/attributes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.attributes import ATTRIBUTE_SETS, attribute_grade  # noqa: E402
from sasa.multi_attribute import (  # noqa: E402
    IDENTIFIABILITY_GATE,
    AttributeSet,
    GradedSubspace,
)
from sasa.probes import Probe, confound_angles  # noqa: E402


def spearman(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Spearman rank correlation with a normal-approximation p-value.

    Implemented in-repo so no SciPy version can move a reported p-value, in the
    same spirit as the Wilcoxon test in ``experiments/run_pilot.py``.

    Args:
        a: First sample.
        b: Second sample, same length.

    Returns:
        A ``(rho, p_value)`` tuple. Returns ``(0.0, 1.0)`` for short samples.

    Raises:
        ValueError: If the inputs differ in length.
    """
    import math

    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 3:
        return 0.0, 1.0

    def ranks(xs: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: xs[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = ranks(list(a)), ranks(list(b))
    mean = (n + 1) / 2.0
    da = [r - mean for r in ra]
    db = [r - mean for r in rb]
    num = sum(x * y for x, y in zip(da, db))
    den = (sum(x * x for x in da) * sum(y * y for y in db)) ** 0.5
    if den == 0:
        return 0.0, 1.0
    rho = num / den
    t = rho * ((n - 2) / (1 - rho ** 2)) ** 0.5 if abs(rho) < 1 else 0.0
    # Two-sided normal tail.
    p = math.erfc(abs(t) / (2 ** 0.5)) if abs(rho) < 1 else 0.0
    return rho, p


@torch.no_grad()
def _states(model, tokenizer, texts: List[str], device) -> torch.Tensor:
    """Return last-token final hidden states for a list of texts.

    Args:
        model: A causal language model.
        tokenizer: The matching tokenizer.
        texts: Texts to encode.
        device: Device to run on.

    Returns:
        Tensor of shape ``(len(texts), d)``.
    """
    out = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=64).to(device)
        res = model(**enc, output_hidden_states=True)
        out.append(res.hidden_states[-1][0, -1, :].cpu())
    return torch.stack(out)


def _retention(
    attrs: AttributeSet,
    base_logits: torch.Tensor,
    token_embeddings: torch.Tensor,
    alpha: float,
) -> Dict[str, float]:
    """How much of each attribute's steering survives the composition.

    For attribute ``i``, compare the change its margin makes to the base logits
    when applied alone against the change when all attributes are applied at the
    same strength. A retention below 1 is interference: another attribute's
    bias has eaten into this one's effect.

    Args:
        attrs: The attribute set.
        base_logits: Unsteered logits, shape ``(B, V)``.
        token_embeddings: Input-embedding matrix, shape ``(V, d)``.
        alpha: Per-attribute strength, identical across attributes so the
            comparison is at matched steering mass.

    Returns:
        A dict of attribute name to retention, plus ``"worst"``.
    """
    solo = {a.name: attrs.compose_logits(
        base_logits, token_embeddings, {a.name: alpha}) for a in attrs.attributes}
    together = attrs.compose_logits(
        base_logits, token_embeddings, {a.name: alpha for a in attrs.attributes}
    )
    out: Dict[str, float] = {}
    for a in attrs.attributes:
        feature = a.token_bias(token_embeddings)
        if float(feature.abs().max()) < 1e-12:
            out[a.name] = float("nan")
            continue
        solo_effect = (solo[a.name] - base_logits) @ feature
        joint_effect = (together - base_logits) @ feature
        denom = solo_effect.abs().mean().clamp_min(1e-12)
        out[a.name] = float((joint_effect.abs().mean() / denom).item())
    finite = [v for v in out.values() if v == v]
    out["worst"] = min(finite) if finite else float("nan")
    return out


def run_interference(
    attrs: AttributeSet, base_logits: torch.Tensor, token_embeddings: torch.Tensor,
    alphas: List[float],
) -> Dict[str, object]:
    """Compare the static prediction against measured retention loss.

    Args:
        attrs: The gated attribute set.
        base_logits: Unsteered logits, shape ``(B, V)``.
        token_embeddings: Input-embedding matrix.
        alphas: Per-attribute strengths to evaluate.

    Returns:
        A dict with the per-pair table, the Spearman test, and the alphas used.
    """
    pairs = attrs.interference()
    rows: List[Dict[str, object]] = []
    for alpha in alphas:
        retention = _retention(attrs, base_logits, token_embeddings, alpha)
        for p in pairs:
            rows.append({
                "alpha": alpha,
                "attribute_a": p.attribute_a,
                "attribute_b": p.attribute_b,
                "predicted_overlap": p.subspace_overlap,
                "retention_a": retention[p.attribute_a],
                "retention_b": retention[p.attribute_b],
                "worst_retention": min(retention[p.attribute_a],
                                       retention[p.attribute_b]),
            })
    # Pool across alphas: mean predicted overlap vs mean retention loss.
    pooled: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for r in rows:
        key = (r["attribute_a"], r["attribute_b"])
        bucket = pooled.setdefault(key, {"overlap": [], "loss": []})
        bucket["overlap"].append(r["predicted_overlap"])
        bucket["loss"].append(1.0 - r["worst_retention"])

    overlaps = [sum(v["overlap"]) / len(v["overlap"]) for v in pooled.values()]
    losses = [sum(v["loss"]) / len(v["loss"]) for v in pooled.values()]
    rho, p_value = spearman(overlaps, losses)
    return {
        "rows": rows,
        "n_pairs": len(pooled),
        "spearman_rho": rho,
        "spearman_p": p_value,
        "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
        "mean_retention_loss": sum(losses) / len(losses) if losses else None,
        "alphas": alphas,
    }


def run_graded_vs_binary(
    model, tokenizer, texts: List[str], grades: List[float], device,
) -> Dict[str, object]:
    """Compare graded supervision against a binary split on the same data.

    Args:
        model: A causal language model.
        tokenizer: The matching tokenizer.
        texts: Texts with externally judged grades.
        grades: The grade for each text.
        device: Device to run on.

    Returns:
        A dict comparing rank, grade accuracy, and side accuracy.
    """
    if not texts:
        return {"n": 0, "note": "no externally judged texts available"}
    states = _states(model, tokenizer, texts, device)
    y = torch.tensor(grades, dtype=states.dtype)

    graded = GradedSubspace.fit(states, y, name="toxicity_graded",
                                 toxic_from=2.0, shrinkage=1e-3, seed=1)
    binary_y = (y >= 2.0).to(states.dtype)
    binary = GradedSubspace.fit(states, binary_y, name="toxicity_binary",
                                toxic_from=0.5, shrinkage=1e-3, seed=1)
    return {
        "n": len(texts),
        "n_distinct_grades": int(torch.unique(y).numel()),
        "graded": graded.summary(),
        "binary": binary.summary(),
        "grade_accuracy_gain": graded.heldout_accuracy - binary.heldout_accuracy,
        "rank_gain": graded.effective_rank - binary.effective_rank,
    }


def main() -> int:
    """Run all three studies and write the results file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--out", default="results/attributes.json")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--judged", default="results/pilot.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device).eval()
    emb = model.get_input_embeddings().weight.detach()
    d = model.config.hidden_size

    # --- Study 1+3: fit the attribute and confound probes --------------------
    print("[1/4] fitting attribute and confound probes")
    subspaces: List[GradedSubspace] = []
    probe_rows: List[Dict[str, object]] = []
    probes: List[Probe] = []
    for name, (intent, pos_texts, neg_texts) in ATTRIBUTE_SETS.items():
        pos = _states(model, tokenizer, pos_texts, device)
        neg = _states(model, tokenizer, neg_texts, device)
        # Positive class = the attribute is present, so it gets the real grade;
        # the negative controls are all grade 0. toxic_from=1.0 therefore
        # splits them exactly.
        ordered_states = torch.cat([neg, pos])
        ordered_grades = [0.0] * len(neg_texts) + [
            attribute_grade(name, t) for t in pos_texts
        ]
        if len(set(ordered_grades)) < 2:
            print(f"      SKIP {name}: grades collapse to {set(ordered_grades)}")
            continue
        sub = GradedSubspace.fit(
            ordered_states, ordered_grades, name=name,
            toxic_from=1.0, shrinkage=1e-3, seed=1,
        )
        subspaces.append(sub)
        probe = Probe.fit(pos, neg, name=name, shrinkage=1e-3, seed=1)
        probes.append(probe)
        probe_rows.append({**probe.as_dict(), "intent": intent})
        print(f"      {name:16s} rank={sub.effective_rank} "
              f"side_acc={sub.heldout_side_accuracy:.3f} "
              f"probe_z={probe.z_vs_chance():.2f} "
              f"identifiable={probe.is_identifiable()}")

    # --- Study 3: confound angles against every other probe ------------------
    print("[2/4] confound angles")
    confound_rows = []
    for target, other in combinations(probes, 2):
        report = confound_angles(target, [other])
        confound_rows.append(report.as_dict())
    angles = [
        (row["target"], row["confounds"][0]["name"],
         row["confounds"][0]["angle_degrees"])
        for row in confound_rows
    ]

    # --- Study 1: static prediction vs measured retention --------------------
    print("[3/4] interference: static prediction vs measured retention")
    all_attrs = AttributeSet(subspaces)
    gated = all_attrs.identifiable(IDENTIFIABILITY_GATE)
    print(f"      gated in: {gated.names}")
    print(f"      gated out: {[a for a in all_attrs.names if a not in gated.names]}")
    if len(gated) < 2:
        print("ERROR: fewer than two identifiable attributes; cannot measure "
              "interference. Report the gate result and stop.")
        payload = {"probes": probe_rows, "confounds": confound_rows,
                   "interference": None,
                   "note": "gate excluded all but one attribute"}
    else:
        with torch.no_grad():
            rows = []
            for text in [t for _i, p, n in ATTRIBUTE_SETS.values() for t in p][:8]:
                enc = tokenizer(text, return_tensors="pt").to(device)
                rows.append(model(**enc).logits[0, -1, :].cpu())
            base_logits = torch.stack(rows)
        interference = run_interference(
            gated, base_logits, emb,
            alphas=[args.alpha / 4, args.alpha / 2, args.alpha, args.alpha * 2],
        )
        print(f"      pairs={interference['n_pairs']} "
              f"spearman rho={interference['spearman_rho']:+.3f} "
              f"p={interference['spearman_p']:.3f} "
              f"mean overlap={interference['mean_overlap']:.3f} "
              f"mean retention loss={interference['mean_retention_loss']:.3f}")
        payload = {"probes": probe_rows, "confounds": confound_rows,
                   "interference": interference}

    # --- Study 2: graded vs binary on the judged corpus ----------------------
    print("[4/4] graded vs binary supervision on the judged corpus")
    judged_path = Path(args.judged)
    judged = []
    if judged_path.exists():
        data = json.loads(judged_path.read_text())
        for cond, rows_ in data.get("conditions", {}).items():
            for r in rows_:
                if r.get("judge_score") is not None:
                    judged.append((r["text"], float(r["judge_score"])))
    if judged:
        texts = [t for t, _ in judged if t]
        grades = [g for t, g in judged if t]
        graded_result = run_graded_vs_binary(model, tokenizer, texts, grades, device)
        print(f"      n={graded_result['n']} "
              f"grades={graded_result['n_distinct_grades']} "
              f"rank {graded_result['binary']['effective_rank']} -> "
              f"{graded_result['graded']['effective_rank']}  "
              f"grade acc {graded_result['binary']['heldout_grade_accuracy']:.3f} -> "
              f"{graded_result['graded']['heldout_grade_accuracy']:.3f}")
    else:
        graded_result = {"n": 0, "note": "no judged generations found"}
        print("      no judged generations available")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"model": args.model, "hidden": d, "vocab": int(emb.shape[0]),
                 "alpha": args.alpha,
                 "identifiability_gate": IDENTIFIABILITY_GATE},
        "attributes": all_attrs.names,
        "gated_in": gated.names,
        "probes": probe_rows,
        "confounds": confound_rows,
        "confound_min_angles": angles,
        "interference": payload.get("interference"),
        "graded_vs_binary": graded_result,
    }, indent=2))
    print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
