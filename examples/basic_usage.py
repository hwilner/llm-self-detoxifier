"""
Basic usage example for SASA (Self-disciplined Autoregressive Sampling).

This example demonstrates how to train a subspace learner and use it for
detoxified text generation.

Note: This example uses synthetic data for demonstration. In practice, you
would use real toxicity datasets like Jigsaw or RealToxicityPrompts.
"""

import torch
from sasa import SubspaceLearner, SASASampler, BaselineSampler


def create_synthetic_data(n_samples=100, embedding_dim=768):
    """
    Create synthetic toxic and non-toxic embeddings for demonstration.
    
    In practice, these would be extracted from a real language model using
    the extract_embeddings_from_model() function.
    
    Args:
        n_samples: Number of samples per class.
        embedding_dim: Dimension of embeddings.
        
    Returns:
        Tuple of (non_toxic_embeddings, toxic_embeddings).
    """
    # Non-toxic embeddings: cluster around positive first dimension
    embeddings_non_toxic = torch.randn(n_samples, embedding_dim) * 0.5
    embeddings_non_toxic[:, 0] += 2.0
    
    # Toxic embeddings: cluster around negative first dimension
    embeddings_toxic = torch.randn(n_samples, embedding_dim) * 0.5
    embeddings_toxic[:, 0] -= 2.0
    
    return embeddings_non_toxic, embeddings_toxic


def main():
    """
    Main demonstration of SASA usage.
    """
    print("SASA Basic Usage Example")
    print("=" * 50)
    
    # Step 1: Create synthetic training data
    print("\n1. Creating synthetic training data...")
    embedding_dim = 768  # Typical for GPT-2 and similar models
    embeddings_non_toxic, embeddings_toxic = create_synthetic_data(
        n_samples=200,
        embedding_dim=embedding_dim
    )
    print(f"   Non-toxic samples: {embeddings_non_toxic.shape}")
    print(f"   Toxic samples: {embeddings_toxic.shape}")
    
    # Step 2: Train subspace learner
    print("\n2. Training subspace learner...")
    learner = SubspaceLearner(embedding_dim=embedding_dim)
    params = learner.fit(embeddings_non_toxic, embeddings_toxic)
    print(f"   Learned weight vector shape: {params.w_v.shape}")
    print(f"   Learned bias vector shape: {params.b_v.shape}")
    
    # Step 3: Test classification
    print("\n3. Testing classification...")
    test_non_toxic = torch.randn(embedding_dim) * 0.5
    test_non_toxic[0] += 2.0
    test_toxic = torch.randn(embedding_dim) * 0.5
    test_toxic[0] -= 2.0
    
    score_non_toxic = learner.classify(test_non_toxic)
    score_toxic = learner.classify(test_toxic)
    
    print(f"   Non-toxic test score: {score_non_toxic.item():.3f} (should be positive)")
    print(f"   Toxic test score: {score_toxic.item():.3f} (should be negative)")
    
    # Step 4: Compute margins
    print("\n4. Computing margins...")
    margin_non_toxic = learner.compute_margin(test_non_toxic)
    margin_toxic = learner.compute_margin(test_toxic)
    
    print(f"   Non-toxic margin: {margin_non_toxic.item():.3f}")
    print(f"   Toxic margin: {margin_toxic.item():.3f}")
    
    # Step 5: Create SASA sampler
    print("\n5. Creating SASA sampler...")
    sampler = SASASampler(
        subspace_learner=learner,
        alpha=1.0,  # Detoxification strength
        temperature=1.0
    )
    print(f"   Alpha (detoxification strength): {sampler.alpha}")
    print(f"   Temperature: {sampler.temperature}")
    
    # Step 6: Demonstrate margin computation for tokens
    print("\n6. Demonstrating token margin computation...")
    vocab_size = 50257  # GPT-2 vocabulary size
    current_embedding = torch.randn(embedding_dim)
    token_embeddings = torch.randn(vocab_size, embedding_dim)
    
    margins = sampler.compute_token_margins(current_embedding, token_embeddings)
    print(f"   Computed margins for {vocab_size} tokens")
    print(f"   Margin range: [{margins.min().item():.3f}, {margins.max().item():.3f}]")
    
    # Step 7: Demonstrate logit adjustment
    print("\n7. Demonstrating logit adjustment...")
    logits = torch.randn(vocab_size)
    adjusted_logits = sampler.adjust_logits(
        logits,
        current_embedding,
        token_embeddings
    )
    
    diff = (adjusted_logits - logits).abs().mean()
    print(f"   Average logit change: {diff.item():.3f}")
    
    # Step 8: Save and load parameters
    print("\n8. Testing save and load...")
    save_path = "/tmp/sasa_params.pt"
    learner.save(save_path)
    print(f"   Saved parameters to {save_path}")
    
    new_learner = SubspaceLearner(embedding_dim=embedding_dim)
    new_learner.load(save_path)
    print("   Loaded parameters into new learner")
    
    # Verify loaded parameters work correctly
    score_loaded = new_learner.classify(test_non_toxic)
    print(f"   Loaded learner score: {score_loaded.item():.3f}")
    print(f"   Original learner score: {score_non_toxic.item():.3f}")
    print(f"   Difference: {abs(score_loaded.item() - score_non_toxic.item()):.6f}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nNext steps:")
    print("- Replace synthetic data with real toxicity datasets")
    print("- Load a real language model (GPT-2, Llama, etc.)")
    print("- Use sampler.generate() to produce detoxified text")


if __name__ == "__main__":
    main()
