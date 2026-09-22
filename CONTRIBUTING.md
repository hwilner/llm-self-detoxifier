# Contributing

Welcome! This repository implements SASA (Ko et al. 2024, arXiv:2410.03818) and is being extended along the roadmap in `docs/ROADMAP.md`. Contributions of all sizes are welcome — no research background is required for many tasks.

## Reading the issue cards

All planned work lives in GitHub issues written in a standard card format. Before starting one, read:

- **Size** — XS (a few hours), S (a day or two), L (multi-week epic; do not pick up directly — pick one of its sub-tasks).
- **Dependencies** — issues that must be completed first. If the card says "Blocked by: … (issue #N)", wait for #N to close or coordinate in the comments.
- **Acceptance criteria** — the exact conditions under which the work counts as done. Meet all of them.
- **Boundary** — what the task explicitly does *not* authorize. Stay inside it; if you think the boundary is wrong, comment on the issue first.

Ready-to-start issues are marked **READY** in `docs/PROJECT.md`.

## Development setup

```bash
git clone https://github.com/hwilner/llm-self-detoxifier.git
cd llm-self-detoxifier
pip install -e .
pytest tests/test_sasa.py -v   # 15 tests should pass
```

Python 3.8+, PyTorch 2.0+, Transformers 4.30+ (see `requirements.txt`).

## Branch and PR conventions

- Branch from `main`: `feature/short-description` or `fix/short-description`.
- One issue per PR; link it ("Closes #N") in the PR description.
- Fill in the PR template: summary, linked issue, test evidence (paste `pytest` output), checklist.
- Keep diffs focused. Do not reformat unrelated code.
- Docs use the citation pack established in `docs/INTRODUCTION.md` / `docs/ROADMAP.md` — cite papers with exact title, authors, year, and arXiv ID.

## Scientific integrity rules

These are non-negotiable for any empirical work in this repo:

1. **Tests pass before merge.** New behavior ships with new tests in `tests/`.
2. **Honest negatives.** If an experiment fails or a number regresses, report it. Deleting or hiding negative results is worse than a bad number.
3. **Pre-registered metrics.** For benchmark runs, fix metrics, alpha values, and significance tests *before* running (see `docs/METHODS.md`, sections U1–U4). No metric-shopping after the fact.
4. **Label automatic-only results.** Toxicity scores from automatic scorers (e.g., Perspective API) are proxies; say so when you report them without human spot-checks.
5. **Stay in scope.** An issue card does not authorize empirical analysis beyond its documented scope.

## Questions

Open an issue, or comment on the card you're working on. Be kind, assume good faith, and have fun.
