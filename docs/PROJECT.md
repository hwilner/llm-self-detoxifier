# Project Board

In-repo mirror of the backlog. The source of truth is the GitHub issue tracker; this file summarizes it for contributors.

## Status: Ready to start

These issues have no unresolved dependencies ("READY"):

| Issue | Title | Size |
|-------|-------|------|
| #4 | Add BOLD bias + sycophancy probe evaluation harness | S |
| #5 | Human spot-check protocol doc + inter-annotator sheet | S |
| #8 | Llama-family hidden-state adapter + tests | S |
| #7 | Alpha scheduling module (fixed/linear/cosine) + tests | S |
| #10 | Expose subspace as serializable module + reproduce SASA on GPT-2-L and Llama-3.1-8B | S |
| #17 | pyproject.toml packaging cleanup + pytest config | XS |
| #16 | Example notebook/script: detoxify GPT-2 on 5 prompts end-to-end | XS |
| #15 | docs/API_REFERENCE.md for SubspaceLearner/SASASampler/BaselineSampler | S |

## Epic: Phase 1 — Evaluation hardening (#1, L)

| Issue | Title | Size | Blocked by |
|-------|-------|------|------------|
| #4 | Add BOLD bias + sycophancy probe evaluation harness | S | — (READY) |
| #5 | Human spot-check protocol doc + inter-annotator sheet | S | — (READY) |
| #6 | Full benchmark re-run with pre-registered metrics + honest report | XS | #4, #5 |

## Epic: Phase 2 — Multi-model support and efficiency (#2, L)

| Issue | Title | Size | Blocked by |
|-------|-------|------|------------|
| #8 | Llama-family hidden-state adapter + tests | S | — (READY) |
| #7 | Alpha scheduling module (fixed/linear/cosine) + tests | S | — (READY) |
| #9 | Latency/overhead benchmark table vs BaselineSampler | XS | #7, #8 |

## Epic: Phase 3 (novel research) — TSM-MA: transferred subspace margins with multi-attribute scheduling (#3, L)

| Issue | Title | Size | Blocked by |
|-------|-------|------|------------|
| #10 | Expose subspace as serializable module + reproduce SASA on GPT-2-L and Llama-3.1-8B | S | — (READY) |
| #11 | Procrustes alignment map between paired hidden states of proxy/target + synthetic tests | S | #10 |
| #12 | Small→large transfer experiment: Llama-3.2-1B subspace → 8B decoding, vs direct-learned baseline | S | #11 |
| #13 | Multi-attribute margin composition (toxicity + sycophancy + bias) with per-attribute alpha schedules + interference study | S | #10 |
| #14 | Results report: transfer gap ≤15% criterion, fluency cost, compute comparison, honest failure modes | S | #12, #13 |

## Standalone (Phase 0 hygiene, no epic)

| Issue | Title | Size | Blocked by |
|-------|-------|------|------------|
| #17 | pyproject.toml packaging cleanup + pytest config | XS | — (READY) |
| #16 | Example notebook/script: detoxify GPT-2 on 5 prompts end-to-end | XS | — (READY) |
| #15 | docs/API_REFERENCE.md for SubspaceLearner/SASASampler/BaselineSampler | S | — (READY) |

## Suggested contribution paths

- **First-time contributor (docs, no ML needed):** #16 (example script), #15 (API reference), #5 (human-eval protocol).
- **First-time contributor (code):** #17 (packaging), #7 (alpha scheduling).
- **ML engineer:** #8 (Llama adapter), #4 (evaluation harness).
- **Researcher:** #10 (reproduction + serializable subspace) is the gateway drug to the TSM-MA chain (#11 → #12 / #13 → #14).

## Rules

- **Tests pass** before merge; new behavior ships with tests (`pytest tests/test_sasa.py -v`).
- **Honest negatives:** failed experiments and regressed numbers are reported, never hidden.
- **Pre-registered metrics** for benchmark runs, fixed in advance per `docs/METHODS.md` (U1–U4).
- **Stay inside the Boundary** section of your issue card; an issue does not authorize work beyond its documented scope.

---

*Note: this in-repo board is a snapshot for contributors. Creating a real GitHub Project board (with custom fields, views, and automation) requires repository-owner permissions and is an owner action.*

## Live GitHub Project

The public [contribution board](https://github.com/users/hwilner/projects/11) mirrors the active issue backlog. **Implementation, tests, documentation, evaluation, and scoped extensions to the original project are welcome from all contributors.** Choose a `Ready` card only after reading its boundary and acceptance criteria; the GitHub issue body remains the source of truth for scope and dependencies.


## Refined contributor child cards

The original broad cards below remain **open parent/integration tasks**. They were not deleted, replaced, closed, or assigned. Each small linked child is an unassigned, focused contribution unit; contributors should claim one child rather than duplicate parent work.

| Parent task | Linked child issue | Focus |
| --- | --- | --- |
| #6 | [#18](https://github.com/hwilner/llm-self-detoxifier/issues/18) | [XS] Frozen benchmark configuration and deterministic rerun command |
| #6 | [#19](https://github.com/hwilner/llm-self-detoxifier/issues/19) | [XS] Benchmark execution and automatic-plus-human evaluation record collection |
| #6 | [#20](https://github.com/hwilner/llm-self-detoxifier/issues/20) | [XS] Honest benchmark results report with signed-rank and bootstrap outputs |
| #10 | [#21](https://github.com/hwilner/llm-self-detoxifier/issues/21) | [XS] Serializable subspace format with save-load round-trip tests |
| #10 | [#22](https://github.com/hwilner/llm-self-detoxifier/issues/22) | [XS] GPT-2-L SASA smoke reproduction with recorded seed and configuration |
| #10 | [#23](https://github.com/hwilner/llm-self-detoxifier/issues/23) | [XS] Llama-3.1-8B SASA smoke reproduction with recorded seed and configuration |
| #12 | [#24](https://github.com/hwilner/llm-self-detoxifier/issues/24) | [XS] Deterministic paired hidden-state collection with provenance |
| #12 | [#25](https://github.com/hwilner/llm-self-detoxifier/issues/25) | [XS] Transfer, direct, and identity-or-PCA baseline decoding runs |
| #12 | [#26](https://github.com/hwilner/llm-self-detoxifier/issues/26) | [XS] Locked statistical and perplexity result table generation |
| #13 | [#27](https://github.com/hwilner/llm-self-detoxifier/issues/27) | [XS] Multi-margin composition API with unit tests |
| #13 | [#28](https://github.com/hwilner/llm-self-detoxifier/issues/28) | [XS] Per-attribute schedule and attribute-margin fixtures |
| #13 | [#29](https://github.com/hwilner/llm-self-detoxifier/issues/29) | [S] Multi-attribute interference-study runner and outputs |
