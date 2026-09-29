# Results

**Generated 2026-09-29 by executing this repository's own test suite.**

This file records what was verified and what was not. It is a record of one
concrete verification run, not a claim about scientific merit.

## Verification run

| Metric | Value |
|---|---|
| Tests passed | **15** |
| Tests failed | 0 |
| Errors | 0 |
| Skipped | 0 |
| Wall clock | 4.27 s |
| Environment | Python 3.11, CPU-only, no GPU |

Reproduce with:

```bash
PYTHONPATH=$PWD:$PWD/src pytest -q
```

## What this does and does not establish

**Established:** the code in this repository imports cleanly and its test suite
passes in the environment above (15 tests, 0 failures). That is real
evidence, and it is the kind of evidence that was previously missing from this
project.

**Not established:**

- No scientific claim is verified by a passing unit test. These tests check that
  functions behave as specified -- not that a biological or computational
  hypothesis holds.
- No benchmark, metric, or comparative result was reproduced. The unit tests are
  not experiments.
- Coverage was not measured. A passing suite can still leave large areas untested.

**No quantitative performance claims were detected in the README**, so there is
no benchmark figure here that this run confirms or contradicts.

## Negative and unverified results

Recorded explicitly, so they are not mistaken for successes:

- **No experimental result was reproduced for this repository.** If the README
  or `docs/` describe findings, they were not re-derived here.
- **The test suite passing is not evidence the research is correct.** It is
  evidence the code is internally consistent.

## Reproducing

```bash
PYTHONPATH=$PWD:$PWD/src pytest -q
```

