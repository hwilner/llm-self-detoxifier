# Results

This document is the single source of truth for what this repository has
**measured**, as opposed to what it implements. Per the project's standing
rules (`docs/METHODS.md`): negative or null results are reported, not deleted;
no number appears here without a stated metric, configuration, and scope.

## Current status at a glance

| Claim | Status |
|---|---|
| SASA reduces toxicity on GPT-2 (RealToxicityPrompts, AttaQ) | Paper-aligned numbers below; **single model, single alpha, automatic metrics only** |
| New features (KV cache, top-k margins, gating, adapters, multi-layer, circuit breaker) | Implemented and logic-verified; **benchmark numbers pending the pre-registered re-run** (Phase 1, issues #18–#20) |
| Mamba/Jamba separability | Probe implemented (`experiments/separability_probe.py`); **no real-model numbers yet** |
| Cross-model transfer (TSM-MA) | Not yet run (Phase 3) |

## Paper-aligned GPT-2 numbers (original evaluation)

These are the numbers previously reported in the README, from the initial
educational implementation. They use automatic toxicity scoring only.

| Benchmark | Baseline toxicity | SASA toxicity | Relative reduction |
|---|---|---|---|
| RealToxicityPrompts | 0.481 | 0.426 | ~10% |
| AttaQ | 0.264 | 0.142 | ~42% |

Perplexity was comparable to baseline (exact ΔPPL not recorded in the original
run — one reason the Phase 1 pre-registered re-run exists).

**Caveats (stated honestly):** single model (GPT-2), single alpha, automatic
toxicity metrics only (gameable via evasive paraphrase — see `docs/METHODS.md`
U1 and the human spot-check protocol, issue #5), no significance testing per
U3 yet. Treat these as a smoke result, not a validated claim.

## Verification results for new code (logic-level)

The following were verified with the shipped code against a tiny
HF-interface model and synthetic data (not benchmarks):

- KV-cache parity: cached and uncached generation produce identical token
  sequences (fixed seed) — `SASASampler` and `BaselineSampler`.
- Top-k margin restriction: tail-token margins exactly zero; top-k tokens
  receive real margins; `margin_top_k=None` reproduces full-vocab behavior.
- Margin-gated alpha: safe contexts decode unmodified; toxic-side contexts
  are steered; `gate_threshold=None` reproduces always-steer behavior.
- Multi-layer learner: recovers the planted most-separable layer on synthetic
  data; ensemble margin equals the mean of per-layer margins; save/load
  round-trip exact.
- Circuit breaker: aborts after exactly k consecutive toxic-side steps;
  silent on safe contexts; off by default.
- `experiments/separability_probe.py --synthetic` runs end-to-end offline.

## Pre-registered criteria for future runs

All future numbers in this file must conform to `docs/METHODS.md`:

- **U1**: Perspective API primary when claiming comparability with published
  SASA/DExperts results; local classifier as secondary.
- **U2**: fixed alpha is the reference; schedules adopted only if they beat it
  by the pre-registered margin.
- **U3**: Wilcoxon signed-rank + prompt-level bootstrap CIs for headline
  comparisons; t-test secondary.
- **U4**: transfer-map selection by held-out margin-preservation error.
- **U5 (utility floor)**: "fluency preserved" requires MAUVE ≥
  baseline-self-MAUVE − 0.05, win-rate ≥ 45% vs baseline, and ΔPPL ≤ +1.

## How to add results

1. Run the frozen benchmark configuration (issue #18) with the deterministic
   rerun command.
2. Add a dated section here with: configuration, seed, metrics per U1–U5,
   significance tests per U3, and honest failure modes.
3. If a result regresses relative to this file, report it anyway.
