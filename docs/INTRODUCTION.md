# Introduction: Why and How SASA Detoxifies Language Models

This document explains, from the ground up, the problem this repository tackles, the line of research it belongs to, how SASA works in plain mathematical language, what results have been reported, and where the field is heading. It is written for a technical reader (comfortable with basic linear algebra and neural networks) who is new to controlled text generation.

![Concept figure: SASA decoding — the prompt's hidden state is located relative to a learned hyperplane separating toxic and non-toxic regions; each candidate token's signed margin is added to its logit (scaled by alpha), yielding a safer next token.](figures/concept_figure.svg)

*Figure 1. SASA in one pass: read the hidden state, measure its margin to the toxic/non-toxic hyperplane for each candidate next token, add alpha times that margin to the logits, and sample — no external models involved.*

## 1. The problem: LLMs sometimes produce toxic text

Large language models (LLMs) are trained on enormous corpora of internet text. That text contains toxicity — insults, slurs, threats, profanity-laced abuse — and LLMs learn to reproduce it. Even when a model is generally well-behaved, certain prompts can reliably coax toxic continuations out of it. Gehman et al. introduced **RealToxicityPrompts** [7], a benchmark of sentence prefixes drawn from web text, and showed that even models with standard mitigations degenerate into toxic generations on a substantial fraction of prompts. Toxicity of generated text is typically scored automatically, most commonly with the Perspective API classifier [7], which returns a probability that text would be perceived as toxic.

The practical question is: *given a frozen, pre-trained model, how do we reduce the probability that it emits toxic text, without retraining it and without wrecking its fluency?* This is an instance of the broader field of **controllable text generation**, surveyed comprehensively by Liang et al. [12].

## 2. The lineage of decoding-time control

SASA is the latest step in a decade-long line of *decoding-time* (inference-time) control methods. Understanding the lineage makes SASA's contribution clear.

**PPLM (Plug and Play Language Models), Dathathri et al., ICLR 2020 [3].** PPLM trains a small attribute classifier (e.g., "is this text toxic?") on top of the frozen LM. At each generation step, gradients from the classifier are used to *perturb the model's hidden states* (the key-value history) so that the continuation scores higher on the desired attribute. It works, but it requires multiple gradient steps per token — expensive — and can degrade fluency because the hidden-state perturbation is a blunt instrument.

**GeDi (Generative Discriminator), Krause et al., Findings of EMNLP 2021 [4].** GeDi trains class-conditional LMs (one conditioned on "toxic", one on "non-toxic") and uses their log-probability ratio as a discriminator that *rescales the base model's logits* at each step. Faster than PPLM, but it requires training auxiliary class-conditional models.

**FUDGE, Yang & Klein, NAACL 2021 [5].** FUDGE trains a lightweight "future discriminator" that predicts, from a partial sequence, whether the *completed* sequence will have the attribute. Its scores adjust the base model's logits. FUDGE is notable for being composable: multiple attribute predictors can be combined at decode time. Still, an external predictor must be trained per attribute.

**DExperts, Liu et al., ACL-IJCNLP 2021 [2].** DExperts combines a fine-tuned "expert" LM (on desirable text) and an "anti-expert" LM (on undesirable text) with the base model: the final token distribution is the product of the base and expert probabilities divided by the anti-expert's. Strong detoxification results, but it requires *two additional fine-tuned language models* in memory at inference time.

**RAD (Reward-Augmented Decoding), Deng & Raffel, EMNLP 2023 [6].** RAD weights tokens by an external reward model's score of the partial sequence, steering generation toward high-reward continuations without fine-tuning. Again, an external model is required.

A related but distinct thread works in *activation space* rather than logit space: Activation Addition (Turner et al. [9]), Representation Engineering (Zou et al. [10]), Inference-Time Intervention (Li et al. [11]), and Contrastive Activation Addition (Rimsky et al. [17]) all *add steering vectors to hidden states* to change behavior. These are powerful but orthogonal to logit-space decoding control, and the two spaces are rarely bridged (see Section 6).

## 3. What makes SASA different

**SASA (Self-disciplined Autoregressive Sampling)**, from "Large Language Models can be Strong Self-Detoxifiers" by Ko, Chen, Das, Mroueh et al. (2024) [1], removes the external model entirely. It is, to our knowledge, the first detoxification method that is:

1. **Purely internal.** The toxicity signal comes from the target model's *own hidden states*. No expert, anti-expert, reward model, or attribute classifier network is needed at decode time.
2. **Linear and closed-form.** The toxic/non-toxic separator is a linear classifier fit in closed form (a Gaussian/Bayes model), not a trained neural network.
3. **Margin-based.** Rather than rescaling logits by a classifier probability, SASA adds the classifier's *signed margin* (distance to the decision boundary) to the logits, with a single strength knob `alpha`.

The intuition: an LLM that was trained to model web text *already knows* — implicitly, in its representations — what toxic text looks like. SASA just finds that knowledge, linearizes it into a direction, and uses it as a compass during decoding.

## 4. The math, in words

**Subspace learning.** Take a labeled dataset of toxic and non-toxic texts. Run each through the frozen LM and record the hidden state of the last token — the *context embedding* `g(c ⊕ x)`. SASA models each class as a multivariate Gaussian: the non-toxic embeddings have some mean and covariance, and so do the toxic ones. Under the Gaussian assumption, the optimal (Bayes) classifier is linear: it is a hyperplane defined by a weight vector `w` and bias `b`, so classification is `sign(wᵀ(g(c⊕x) − b))`. There is a closed-form solution for `w` and `b` in terms of the class means and pooled covariance — essentially Fisher's linear discriminant. In this repo, `SubspaceLearner` (`sasa/subspace_learner.py`) implements this fitting, and exposes `compute_margin`, which returns the signed distance `wᵀ(g − b)`: positive means the context sits on the non-toxic side of the boundary, negative means the toxic side. The magnitude is the model's "confidence," expressed as distance to the boundary rather than a probability.

**Margin-based sampling.** At each decoding step, the LM produces logits over the vocabulary. SASA asks: *if we appended each candidate token, where would the new context land relative to the toxic subspace?* `SASASampler.compute_token_margins` (`sasa/sampler.py`) approximates the next context embedding for every candidate token and computes its margin, producing a margin value per vocabulary token. Then the core update:

```
logits_adjusted = logits + alpha * margins
```

Tokens whose continuation would push the context *away* from the toxic subspace get a boost; tokens that would push *toward* it get suppressed. The softmax over the adjusted logits defines the sampling distribution. This update is not ad hoc: in the paper it falls out of a constrained optimization that maximizes the margin (alignment) subject to staying close to the original distribution (utility), with `alpha` playing the role of the trade-off weight [1].

**Baselines.** `BaselineSampler` (`sasa/sampler.py`) performs ordinary autoregressive sampling with the same temperature/top-k/top-p options, so every reported number can be compared against an identical pipeline minus the margin term.

## 5. Reported results, honestly framed

Using this implementation with GPT-2, the repository reports:

| Benchmark | Baseline toxicity | SASA toxicity | Relative reduction |
|-----------|-------------------|---------------|--------------------|
| RealToxicityPrompts [7] | 0.481 | 0.426 | ~10% |
| AttaQ | 0.264 | 0.142 | ~42% |

Fluency (perplexity) is reported as comparable to baseline. Three caveats deserve emphasis:

- **Automatic toxicity scores are proxies.** Perspective API-style scores can be gamed by evasive paraphrase, and they embed their own biases. Human spot-checks are planned (see `docs/ROADMAP.md` Phase 1).
- **Numbers are single-configuration.** One model, one `alpha`. Sensitivity to `alpha`, layer choice, and dataset size is not yet mapped.
- **"10% reduction" is not "solved."** On RealToxicityPrompts, SASA is a nudge in the right direction, not a guarantee. These results are best read as evidence that internal subspaces carry usable detoxification signal at near-zero cost.

## 6. Limitations

- **Single attribute.** SASA as published handles one binary attribute (toxic vs non-toxic). The paper explicitly defers multi-attribute composition to future work [1].
- **Per-model learning.** The subspace is learned from a given model's own hidden states; there is no claim it transfers to a different architecture.
- **Approximation in this implementation.** `compute_token_margins` approximates the next context embedding as an average of the current embedding and the token embedding, rather than running the full transformer forward pass per candidate. This keeps decode-time cost low but is a simplification of the paper's procedure.
- **Linear separability assumption.** Toxicity is not guaranteed to be linearly separable in any given layer's hidden states; the Gaussian assumption is a modeling convenience.

## 7. Where the field is going

Three trends converge on the open problems SASA leaves behind:

1. **Activation steering is maturing.** ActAdd [9], RepE [10], ITI [11], and CAA [17] show that simple vector arithmetic on hidden states can control behavior. Contrastive decoding [8] similarly exploits differences between models' logits. A natural question is whether logit-space margin control (SASA) and activation-space steering can be unified or combined.
2. **Cross-model transfer is emerging.** The Platonic Representation Hypothesis [13] argues that sufficiently capable models converge toward similar internal representations, which would make transfer feasible. Empirically, Huang et al. [14] transfer concept steering vectors between LLMs with learned linear maps (including weak-to-strong), and Oozeer et al. [15] show refusal/backdoor activation interventions transfer across Llama, Qwen, and Gemma families. But *no published work transfers a toxicity subspace for margin-based decoding* — that gap motivates the TSM-MA direction described in `docs/ROADMAP.md`.
3. **Multi-attribute composition.** FUDGE composes predictors [5]; concept algebra composes subspaces in diffusion models [16]. Composing *margins in logit space* — toxicity plus bias plus sycophancy, each with its own schedule — is unclaimed territory [1].

## References

1. Ko, Chen, Das, Mroueh, Dan, Kollias, Chaudhury, Pedapati, Daniel. *Large Language Models can be Strong Self-Detoxifiers.* 2024. arXiv:2410.03818.
2. Liu et al. *DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts.* ACL-IJCNLP 2021. arXiv:2105.03023.
3. Dathathri et al. *Plug and Play Language Models: a Simple Approach to Controlled Text Generation.* ICLR 2020. arXiv:1912.02164.
4. Krause et al. *GeDi: Generative Discriminator Guided Sequence Generation.* Findings of EMNLP 2021. arXiv:2009.06367.
5. Yang & Klein. *FUDGE: Controlled Text Generation With Future Discriminators.* NAACL 2021. arXiv:2104.05218.
6. Deng & Raffel. *Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model.* EMNLP 2023. arXiv:2310.09520.
7. Gehman et al. *RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models.* Findings of EMNLP 2020. arXiv:2009.11462.
8. Li et al. *Contrastive Decoding: Open-ended Text Generation as Optimization.* ACL 2023. arXiv:2210.15097.
9. Turner et al. *Activation Addition: Steering Language Models Without Optimization.* 2023. arXiv:2308.10248.
10. Zou et al. *Representation Engineering: A Top-Down Approach to AI Transparency.* 2023. arXiv:2310.01405.
11. Li et al. *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.* NeurIPS 2023. arXiv:2306.03341.
12. Liang et al. *Controllable Text Generation for Large Language Models: A Survey.* 2024. arXiv:2408.12599.
13. Huh, Cheung, Wang, Isola. *The Platonic Representation Hypothesis.* ICML 2024. arXiv:2405.07987.
14. Huang et al. *Cross-model transfer of concept steering vectors.* ACL 2025. arXiv:2501.02009.
15. Oozeer et al. *Activation Space Interventions Can Be Transferred Between Large Language Models.* ICML 2025. arXiv:2503.04429.
16. Trager et al. *Linear Spaces of Meanings: Compositional Structures in Vision-Language Models (Concept Algebra).* NeurIPS 2023. arXiv:2302.03693.
17. Rimsky et al. *Steering Llama 2 via Contrastive Activation Addition.* 2023. arXiv:2312.06681.
