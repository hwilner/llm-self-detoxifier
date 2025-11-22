# SASA: Self-disciplined Autoregressive Sampling for LLM Detoxification

An implementation of **SASA (Self-disciplined Autoregressive Sampling)** from the paper "Large Language Models can be Strong Self-Detoxifiers" by Ko et al. (2024). This repository provides a lightweight, training-free method to reduce toxic output generation in large language models.

## Overview

SASA is a controlled decoding algorithm that leverages the internal representations of language models to steer text generation away from toxic content. Unlike existing approaches, SASA requires no external reward models, no retraining, and no fine-tuning. It operates purely at inference time by learning linear subspaces that characterize toxic versus non-toxic content.

### Key Features

**Training-Free Detoxification:** SASA operates entirely at inference time without requiring model retraining or fine-tuning. This makes it practical for deployment with large-scale models where retraining is computationally prohibitive.

**No External Models Required:** Unlike methods such as RAD or DExperts that rely on external reward models or classifiers, SASA uses only the target model's internal representations. This reduces computational overhead and simplifies deployment.

**Theoretically Grounded:** The sampling strategy is derived from a constrained optimization problem that balances toxicity reduction (alignment) with maintaining fluency (utility). The solution is provably optimal for the given objective.

**Model Agnostic:** SASA can be applied to any autoregressive language model that exposes hidden states, including GPT-2, Llama, and other transformer-based architectures.

## How It Works

SASA operates in two stages: subspace learning and margin-based sampling.

### Subspace Learning

SASA learns a linear classifier that separates toxic and non-toxic content in the embedding space. Given a dataset of prompt-response pairs labeled as toxic or non-toxic, SASA models each class using Gaussian distributions and derives an optimal Bayes classifier in analytical form.

The classifier takes the form **f(c,x) = sign(w^T(g(c⊕x) - b))**, where **g(c⊕x)** represents the context embedding (the hidden state of the last token), **w** is the weight vector, and **b** is the bias. This classifier defines a hyperplane that separates the toxic and non-toxic regions in the embedding space.

### Margin-Based Sampling

During text generation, SASA computes the margin from the current context to the toxic subspace. The margin represents the signed distance to the decision boundary, with positive values indicating non-toxic content and negative values indicating toxic content.

SASA adjusts the sampling distribution by incorporating this margin information. Specifically, the adjusted logits are computed as **logits_adjusted = logits + alpha * margins**, where **alpha** controls the strength of detoxification. This approach dynamically steers the generation process away from toxic content while maintaining fluency.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/hwilner/llm-self-detoxifier.git
cd llm-self-detoxifier
pip install -e .
```

### Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Transformers 4.30 or higher
- NumPy 1.24 or higher
- Datasets 2.14 or higher

## Quick Start

### Training the Subspace Learner

First, train the subspace learner using labeled prompt-response pairs:

```python
import torch
from transformers import AutoModel, AutoTokenizer
from sasa import SubspaceLearner, extract_embeddings_from_model

# Load model and tokenizer
model = AutoModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare your data (lists of toxic and non-toxic texts)
non_toxic_texts = ["Hello, how are you?", "That's a great idea!", ...]
toxic_texts = ["I hate you", "You're stupid", ...]

# Extract embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embeddings_non_toxic = extract_embeddings_from_model(
    model, tokenizer, non_toxic_texts, device
)
embeddings_toxic = extract_embeddings_from_model(
    model, tokenizer, toxic_texts, device
)

# Train subspace learner
learner = SubspaceLearner(embedding_dim=model.config.hidden_size)
learner.fit(embeddings_non_toxic, embeddings_toxic)

# Save for later use
learner.save("subspace_params.pt")
```

### Generating Text with SASA

Use the trained subspace learner to generate detoxified text:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sasa import SubspaceLearner, SASASampler

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load subspace learner
learner = SubspaceLearner(embedding_dim=model.config.hidden_size)
learner.load("subspace_params.pt")

# Create SASA sampler
sampler = SASASampler(
    subspace_learner=learner,
    alpha=1.0,  # Detoxification strength
    temperature=1.0
)

# Generate text
prompt = "I think you are"
result = sampler.generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_length=50,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

print(result['text'])
```

## Performance

Based on the original paper, SASA achieves significant toxicity reduction while maintaining fluency:

| Benchmark | Baseline Toxicity | SASA Toxicity | Reduction |
|-----------|-------------------|---------------|-----------|
| RealToxicityPrompts | 0.481 | 0.426 | 10% |
| AttaQ | 0.264 | 0.142 | 42% |

SASA maintains comparable perplexity to baseline models, indicating that fluency is preserved while toxicity is reduced.

## Architecture

The implementation consists of three main components:

**SubspaceLearner** learns the toxic/non-toxic subspace from labeled embeddings. It estimates class-conditional Gaussian parameters and derives the optimal Bayes classifier in closed form. The learned parameters can be saved and reused across multiple generation sessions.

**SASASampler** implements the margin-based sampling algorithm. It computes margins for candidate tokens and adjusts the sampling distribution to steer generation away from toxic content. The sampler supports standard decoding options such as top-k and nucleus sampling.

**BaselineSampler** provides standard autoregressive sampling without SASA for comparison purposes. This allows for direct evaluation of SASA's effectiveness in reducing toxicity.

## Testing

The repository includes comprehensive unit and integration tests. Tests use publicly available toxicity datasets for validation. All tests verify correctness of subspace learning, margin computation, and sampling behavior.

Run the test suite:

```bash
pytest tests/test_sasa.py -v
```

All 15 tests pass, covering initialization, fitting, classification, margin computation, sampling, and save/load functionality.

## Documentation

Additional documentation is available in the `docs/` directory:

- **ARCHITECTURE.md**: Detailed technical architecture and design decisions
- **TESTING.md**: Testing methodology and dataset information

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{ko2024large,
  title={Large Language Models can be Strong Self-Detoxifiers},
  author={Ko, Ching-Yun and Chen, Pin-Yu and Das, Payel and Mroueh, Youssef and Dan, Soham and Kollias, Georgios and Chaudhury, Subhajit and Pedapati, Tejaswini and Daniel, Luca},
  journal={arXiv preprint arXiv:2410.03818},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This implementation is based on the paper "Large Language Models can be Strong Self-Detoxifiers" by Ko et al. (2024). The original research was conducted at IBM Research and MIT.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for bug fixes, improvements, or new features.

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.
