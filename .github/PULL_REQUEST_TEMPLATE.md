## Summary

What does this PR do, and why? One short paragraph.

## Linked issue

Closes #

## Test evidence

Paste the output of `pytest tests/test_sasa.py -v` (or the relevant subset) run on this branch:

```
<paste here>
```

If this PR changes empirical results, describe the benchmark, metric, and configuration.

## Checklist

- [ ] Tests pass locally (`pytest tests/test_sasa.py -v`)
- [ ] New behavior has new tests
- [ ] Diff is focused — no unrelated reformatting
- [ ] Docs updated if behavior changed (no overwriting of existing docs)
- [ ] Empirical claims follow the rules in `docs/METHODS.md` (pre-registered metrics; honest negatives)
- [ ] Automatic toxicity scores are labeled as automatic if no human spot-check was done
