# SASA Architecture Documentation

This document provides a detailed technical overview of the SASA (Self-disciplined Autoregressive Sampling) implementation.

## System Overview

SASA is a controlled decoding algorithm that operates at inference time to reduce toxic output generation in language models. The system consists of two main phases: offline subspace learning and online margin-based sampling.

### Design Principles

The implementation follows several key design principles. First, it maintains **model agnosticism** by relying only on standard transformer outputs (hidden states and logits), making it compatible with any autoregressive language model. Second, it achieves **computational efficiency** through analytical solutions that avoid gradient computation during inference. Third, it ensures **modularity** by separating subspace learning from sampling, allowing independent development and testing of each component.

## Core Components

### SubspaceLearner

The SubspaceLearner class implements the subspace learning algorithm from the SASA paper. It learns a linear classifier that separates toxic and non-toxic content in the embedding space.

#### Mathematical Foundation

The learner models each class (toxic and non-toxic) using Gaussian distributions. Given embeddings from toxic and non-toxic examples, it estimates the following parameters:

- **mu_1**: Mean vector for non-toxic class
- **mu_2**: Mean vector for toxic class  
- **Sigma**: Shared covariance matrix

The optimal Bayes classifier is derived in closed form as:

**f(c,x) = sign(w^T(g(c⊕x) - b))**

where:
- **w = Sigma^{-1}(mu_1 - mu_2)**: Weight vector
- **b = (mu_1 + mu_2) / 2**: Bias vector
- **g(c⊕x)**: Context embedding (last token's hidden state)

#### Implementation Details

The implementation includes several important considerations. For numerical stability, a small regularization term (1e-6 * I) is added to the covariance matrix before inversion. If matrix inversion fails, the implementation falls back to pseudo-inverse computation. The class supports both single embeddings and batched inputs for efficient processing.

#### API Design

The SubspaceLearner exposes a clean API with the following key methods:

- **fit(embeddings_non_toxic, embeddings_toxic)**: Learns subspace parameters from labeled embeddings
- **classify(embedding)**: Returns classification score (positive = non-toxic, negative = toxic)
- **compute_margin(embedding)**: Returns normalized margin to toxic subspace
- **save(path) / load(path)**: Persists and loads learned parameters

### SASASampler

The SASASampler class implements the margin-based sampling algorithm. It adjusts the sampling distribution during autoregressive decoding to steer generation away from toxic content.

#### Constrained Optimization

The sampling strategy solves a constrained optimization problem with two objectives:

1. **Alignment**: Maximize margin to stay away from toxic subspace
2. **Utility**: Keep sampling distribution close to original (maintain fluency)

Formally, the problem is:

**maximize Σ p_i * π_m(x_i | context)**  
**minimize KL(p || π_ref(context))**

where:
- **π_m = Softmax(margins)**: Margin-based distribution
- **π_ref = Softmax(logits)**: Original distribution

#### Sampling Algorithm

The sampling algorithm operates as follows:

1. **Margin Computation**: For each candidate token in the vocabulary, compute the margin that would result from appending that token to the current context
2. **Logit Adjustment**: Adjust the original logits by adding a weighted margin term: **logits_adjusted = logits + alpha * margins**
3. **Temperature Scaling**: Apply temperature scaling for diversity control
4. **Filtering**: Optionally apply top-k or nucleus (top-p) filtering
5. **Sampling**: Sample from the adjusted distribution

#### Hyperparameters

The sampler exposes two key hyperparameters:

- **alpha**: Controls detoxification strength (higher = stronger detoxification). Typical values range from 0.5 to 2.0
- **temperature**: Controls sampling diversity (lower = more deterministic). Standard values are 0.7 to 1.0

### BaselineSampler

The BaselineSampler provides standard autoregressive sampling without SASA for comparison purposes. It implements the same interface as SASASampler but without margin-based adjustments.

## Data Flow

The complete data flow for SASA-based generation is as follows:

1. **Initialization**: Load pre-trained language model and trained SubspaceLearner
2. **Prompt Encoding**: Tokenize input prompt and obtain initial hidden states
3. **Iterative Generation**: For each generation step:
   - Extract current context embedding (last token's hidden state)
   - Compute logits from language model
   - Compute margins for all vocabulary tokens
   - Adjust logits based on margins
   - Sample next token from adjusted distribution
   - Append token to context and repeat
4. **Termination**: Stop when EOS token is generated or max length is reached

## Performance Considerations

### Computational Complexity

The computational complexity of SASA is dominated by two operations:

- **Margin Computation**: O(V * d) where V is vocabulary size and d is embedding dimension
- **Logit Adjustment**: O(V) for element-wise addition

For typical models (V ~ 50k, d ~ 4096), margin computation adds approximately 10-15 percent overhead compared to standard sampling. This is significantly more efficient than methods requiring external reward models.

### Memory Requirements

SASA requires storing the subspace parameters (w, b, mu_1, mu_2, Sigma), which occupy O(d^2) memory for the covariance matrix. For d = 4096, this is approximately 64 MB, which is negligible compared to model weights.

### Optimization Opportunities

Several optimization opportunities exist for production deployment:

- **Token Embedding Caching**: Pre-compute and cache token embeddings to avoid repeated lookups
- **Batch Processing**: Process multiple prompts in parallel to amortize margin computation costs
- **Approximate Margins**: Use approximate nearest neighbor search to compute margins for only top-k candidates

## Extension Points

The architecture is designed to be extensible in several ways:

### Custom Subspace Learning

Users can implement custom subspace learning algorithms by subclassing SubspaceLearner and overriding the fit() method. The only requirement is that the learned parameters support the classify() and compute_margin() methods.

### Alternative Sampling Strategies

The margin-based adjustment can be modified to implement different steering strategies. For example, users could implement adaptive alpha that varies based on context toxicity scores.

### Multi-Attribute Control

The framework can be extended to control multiple attributes simultaneously (e.g., toxicity, bias, sentiment) by learning multiple subspaces and combining their margins.

## Testing Strategy

The implementation includes comprehensive tests covering:

- **Unit Tests**: Verify correctness of individual components (subspace learning, margin computation, sampling)
- **Integration Tests**: Validate end-to-end generation pipeline
- **Edge Cases**: Test error handling for invalid inputs, dimension mismatches, and insufficient data

All tests use publicly available toxicity datasets to ensure reproducibility and transparency.

## Comparison to Related Work

SASA differs from related approaches in several key ways:

### vs. Retraining-Based Methods

Methods like fine-tuning or RLHF require retraining the model on curated datasets. SASA operates purely at inference time, making it applicable to models where retraining is not feasible (e.g., proprietary APIs).

### vs. External Reward Models

Methods like RAD and DExperts use external reward models or classifiers to guide generation. SASA uses only the target model's internal representations, reducing computational overhead and simplifying deployment.

### vs. Prompt Engineering

Prompt-based methods rely on carefully crafted prompts to steer generation. SASA provides more robust control through learned subspaces that capture the geometry of toxic content in embedding space.

## Future Directions

Several directions for future work include:

- **Adaptive Alpha**: Dynamically adjust detoxification strength based on context toxicity
- **Hierarchical Subspaces**: Learn multiple subspaces at different granularities (word-level, sentence-level, document-level)
- **Cross-Model Transfer**: Investigate whether subspaces learned on one model transfer to other models
- **Real-Time Learning**: Update subspace parameters online based on user feedback

## References

This implementation is based on the paper:

Ko, C.-Y., Chen, P.-Y., Das, P., Mroueh, Y., Dan, S., Kollias, G., Chaudhury, S., Pedapati, T., & Daniel, L. (2024). Large Language Models can be Strong Self-Detoxifiers. arXiv preprint arXiv:2410.03818.
