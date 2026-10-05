# Diffusion-LM margin steering (issue #42, research track scaffold)

SASA steers autoregressive decoding with a next-token margin. Masked-diffusion
language models (LLaDA, MDLM, Dream) have no left-to-right next token, so SASA
does not apply directly. This document defines the adaptation and its experiment
plan. **Status: scaffold only — no implementation yet.**

## Generative process recap

A masked-diffusion LM generates by iteratively denoising a partially masked
sequence: at step `t`, the model predicts logits for every masked position,
a remasking/unmasking schedule commits some predictions, and the loop repeats
until no masks remain. Conditioning is bidirectional.

## Adaptation: margin-biased denoising

The SASA analogue operates at each denoising step instead of each token:

1. **Sequence embedding.** Compute a context embedding of the current
   partially-masked sequence (e.g., mean-pooled last-layer hidden states over
   unmasked positions, or a designated pooling position).
2. **Margin.** Compute the margin of that embedding against the toxic
   subspace, exactly as in AR SASA (`SubspaceLearner.compute_margin`).
3. **Logit bias on masked positions.** For each masked position `i` and
   candidate token `v`, add `alpha * margin_i(v)` to the predicted logits,
   where `margin_i(v)` approximates the margin change if `v` were committed at
   position `i` (same additive approximation as AR SASA's
   `compute_token_margins`).
4. **Schedule interaction.** The bias interacts with the unmasking schedule:
   positions near the boundary can be *deprioritized* for commitment this
   step, letting later, better-informed steps resolve them.

This is classifier-free-guidance-style steering with SASA margins instead of a
classifier gradient. It is a **new algorithm**, not a port.

## Interface

A diffusion LM does not satisfy `BackboneAdapter.supports_incremental()`.
It needs a sibling protocol (not yet implemented):

```python
class DiffusionAdapter(Protocol):
    def step_logits(self, masked_ids, step: int) -> Tensor      # (batch, seq, vocab)
    def sequence_embedding(self, masked_ids) -> Tensor          # (embedding_dim,)
    def supports_incremental(self) -> bool: return False
```

`SASALogitsProcessor`-equivalent: a `SASADenoisingBias` applied to
`step_logits` at masked positions only.

## Experiment plan (pre-register before running)

1. Baseline: LLaDA-8B (or MDLM-small for iteration speed) unsteered toxicity
   on RealToxicityPrompts-style prompts adapted for infilling.
2. Margin-biased denoising at alpha grid; record toxicity + fluency (PPL of
   an AR reference model on the output; MAUVE per `docs/METHODS.md` U5).
3. Ablations: embedding source (mean-pool vs last position vs CLS-equivalent),
   bias only on positions committed this step vs all masked positions,
   schedule deprioritization on/off.
4. Success criteria and honest-negative rules mirror Phase 3.

## Failure modes (anticipated)

- Bidirectional conditioning may dilute the margin signal (toxic content can
  be "voted in" from both sides).
- The additive next-embedding approximation is less justified without a
  causal hidden state; a per-position learned probe may be required.
- Schedules that commit many tokens per step reduce steering granularity.
