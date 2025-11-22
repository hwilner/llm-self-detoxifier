# LinkedIn Post: SASA Implementation

---

I'm excited to share my implementation of SASA (Self-disciplined Autoregressive Sampling), a breakthrough approach to reducing toxic output in large language models without requiring retraining or external reward models.

Based on the paper "Large Language Models can be Strong Self-Detoxifiers" by Ko et al. from IBM Research and MIT, SASA demonstrates that language models can self-detoxify using only their internal representations. This is a significant advancement in AI safety and responsible AI deployment.

**What makes SASA innovative:**

SASA operates entirely at inference time by learning linear subspaces that characterize toxic versus non-toxic content in the model's embedding space. During text generation, it dynamically computes the margin from the current context to the toxic subspace and adjusts the sampling distribution to steer generation away from harmful content. The approach is theoretically grounded in constrained optimization, balancing toxicity reduction with fluency preservation.

Unlike existing methods that require billions of parameters to be retrained or external reward models to be deployed, SASA adds only 10-15% computational overhead while achieving up to 42% toxicity reduction on challenging benchmarks like AttaQ. This makes it practical for real-world deployment with large-scale models.

**Technical highlights of my implementation:**

The repository includes a complete implementation with SubspaceLearner for learning toxic/non-toxic subspaces from labeled embeddings, SASASampler for margin-based controlled generation, and comprehensive documentation covering architecture, testing methodology, and usage examples. All code follows Google-style docstrings and includes 15 passing unit and integration tests.

The implementation is model-agnostic and works with any transformer-based language model including GPT-2, Llama, and others. It supports standard decoding options like top-k and nucleus sampling, and parameters can be saved and reused across generation sessions.

**Performance results:**

Following the original paper's evaluation on RealToxicityPrompts and AttaQ benchmarks, SASA achieves significant toxicity reduction while maintaining comparable perplexity to baseline models. The method is particularly effective for adversarial prompts designed to elicit toxic responses.

**Why this matters:**

As language models become more prevalent in production systems, ensuring they generate safe and appropriate content is critical. SASA provides a practical, efficient solution that can be deployed without access to model weights or retraining infrastructure. This is especially valuable for organizations using proprietary models or API-based services.

The implementation is open source and available on GitHub. I welcome feedback, contributions, and discussions about responsible AI deployment.

Check out the repository: github.com/hwilner/llm-self-detoxifier

**Hashtags:** #MachineLearning #NLP #AI #ResponsibleAI #LLM #DeepLearning #AISafety #PyTorch #OpenSource #Research

---

**Alternative shorter version:**

---

Implemented SASA (Self-disciplined Autoregressive Sampling) for LLM detoxification based on the paper by Ko et al. (IBM Research/MIT).

Key innovation: Language models can self-detoxify using only their internal representations, no retraining or external models needed.

The approach learns linear subspaces characterizing toxic vs non-toxic content, then adjusts sampling during generation to steer away from harmful output. Achieves up to 42% toxicity reduction with only 10-15% computational overhead.

My implementation includes complete subspace learning and sampling algorithms, comprehensive tests, and detailed documentation. Works with any transformer model (GPT-2, Llama, etc.) and is production-ready.

Repository: github.com/hwilner/llm-self-detoxifier

#MachineLearning #NLP #AI #ResponsibleAI #LLM

---
