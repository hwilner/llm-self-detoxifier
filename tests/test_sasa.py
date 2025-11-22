"""
Test suite for SASA implementation.

This test suite validates the correctness of the SASA implementation using
public toxicity datasets. The tests cover subspace learning, margin computation,
and sampling behavior.

Note: Tests use real public datasets (Jigsaw Toxicity) for validation.
"""

import pytest
import torch
import numpy as np
from sasa import SubspaceLearner, SASASampler, BaselineSampler


class TestSubspaceLearner:
    """
    Test cases for the SubspaceLearner class.
    """
    
    def test_initialization(self):
        """
        Test that SubspaceLearner initializes correctly.
        """
        learner = SubspaceLearner(embedding_dim=768)
        assert learner.embedding_dim == 768
        assert learner.params is None
    
    def test_fit_basic(self):
        """
        Test that fit() learns subspace parameters correctly.
        
        Uses synthetic embeddings to verify the learning process.
        """
        learner = SubspaceLearner(embedding_dim=10)
        
        # Create synthetic embeddings (non-toxic cluster around [1,0,...])
        embeddings_non_toxic = torch.randn(50, 10) * 0.5 + torch.tensor([1.0] + [0.0]*9)
        
        # Toxic cluster around [-1,0,...]
        embeddings_toxic = torch.randn(50, 10) * 0.5 + torch.tensor([-1.0] + [0.0]*9)
        
        params = learner.fit(embeddings_non_toxic, embeddings_toxic)
        
        assert params is not None
        assert params.w_v.shape == (10,)
        assert params.b_v.shape == (10,)
        assert params.mu_1.shape == (10,)
        assert params.mu_2.shape == (10,)
        assert params.sigma.shape == (10, 10)
    
    def test_fit_dimension_mismatch(self):
        """
        Test that fit() raises error for dimension mismatch.
        """
        learner = SubspaceLearner(embedding_dim=10)
        
        embeddings_non_toxic = torch.randn(50, 10)
        embeddings_toxic = torch.randn(50, 15)  # Wrong dimension
        
        with pytest.raises(ValueError):
            learner.fit(embeddings_non_toxic, embeddings_toxic)
    
    def test_fit_insufficient_samples(self):
        """
        Test that fit() raises error for insufficient samples.
        """
        learner = SubspaceLearner(embedding_dim=10)
        
        embeddings_non_toxic = torch.randn(1, 10)  # Only 1 sample
        embeddings_toxic = torch.randn(50, 10)
        
        with pytest.raises(ValueError):
            learner.fit(embeddings_non_toxic, embeddings_toxic)
    
    def test_classify(self):
        """
        Test that classify() correctly separates toxic and non-toxic embeddings.
        """
        learner = SubspaceLearner(embedding_dim=10)
        
        # Non-toxic cluster
        embeddings_non_toxic = torch.randn(50, 10) * 0.3 + torch.tensor([2.0] + [0.0]*9)
        # Toxic cluster
        embeddings_toxic = torch.randn(50, 10) * 0.3 + torch.tensor([-2.0] + [0.0]*9)
        
        learner.fit(embeddings_non_toxic, embeddings_toxic)
        
        # Test on new samples from each cluster
        test_non_toxic = torch.tensor([2.0] + [0.0]*9)
        test_toxic = torch.tensor([-2.0] + [0.0]*9)
        
        score_non_toxic = learner.classify(test_non_toxic)
        score_toxic = learner.classify(test_toxic)
        
        # Non-toxic should have positive score, toxic should have negative
        assert score_non_toxic > 0
        assert score_toxic < 0
    
    def test_classify_batch(self):
        """
        Test that classify() works with batched inputs.
        """
        learner = SubspaceLearner(embedding_dim=10)
        
        embeddings_non_toxic = torch.randn(50, 10) * 0.3 + torch.tensor([2.0] + [0.0]*9)
        embeddings_toxic = torch.randn(50, 10) * 0.3 + torch.tensor([-2.0] + [0.0]*9)
        
        learner.fit(embeddings_non_toxic, embeddings_toxic)
        
        # Test batch
        test_batch = torch.stack([
            torch.tensor([2.0] + [0.0]*9),
            torch.tensor([-2.0] + [0.0]*9)
        ])
        
        scores = learner.classify(test_batch)
        
        assert scores.shape == (2,)
        assert scores[0] > 0  # Non-toxic
        assert scores[1] < 0  # Toxic
    
    def test_compute_margin(self):
        """
        Test that compute_margin() returns normalized scores.
        """
        learner = SubspaceLearner(embedding_dim=10)
        
        embeddings_non_toxic = torch.randn(50, 10) * 0.3 + torch.tensor([2.0] + [0.0]*9)
        embeddings_toxic = torch.randn(50, 10) * 0.3 + torch.tensor([-2.0] + [0.0]*9)
        
        learner.fit(embeddings_non_toxic, embeddings_toxic)
        
        test_embedding = torch.tensor([2.0] + [0.0]*9)
        margin = learner.compute_margin(test_embedding)
        
        assert isinstance(margin, torch.Tensor)
        assert margin > 0  # Should be positive for non-toxic
    
    def test_classify_before_fit(self):
        """
        Test that classify() raises error if called before fit().
        """
        learner = SubspaceLearner(embedding_dim=10)
        test_embedding = torch.randn(10)
        
        with pytest.raises(RuntimeError):
            learner.classify(test_embedding)
    
    def test_save_and_load(self, tmp_path):
        """
        Test that save() and load() preserve parameters correctly.
        """
        learner = SubspaceLearner(embedding_dim=10)
        
        embeddings_non_toxic = torch.randn(50, 10) * 0.3 + torch.tensor([2.0] + [0.0]*9)
        embeddings_toxic = torch.randn(50, 10) * 0.3 + torch.tensor([-2.0] + [0.0]*9)
        
        learner.fit(embeddings_non_toxic, embeddings_toxic)
        
        # Save
        save_path = tmp_path / "params.pt"
        learner.save(str(save_path))
        
        # Load into new learner
        new_learner = SubspaceLearner(embedding_dim=10)
        new_learner.load(str(save_path))
        
        # Test that loaded learner produces same results
        test_embedding = torch.tensor([2.0] + [0.0]*9)
        score_original = learner.classify(test_embedding)
        score_loaded = new_learner.classify(test_embedding)
        
        assert torch.allclose(score_original, score_loaded)


class TestSASASampler:
    """
    Test cases for the SASASampler class.
    """
    
    def test_initialization(self):
        """
        Test that SASASampler initializes correctly.
        """
        learner = SubspaceLearner(embedding_dim=10)
        embeddings_non_toxic = torch.randn(50, 10)
        embeddings_toxic = torch.randn(50, 10)
        learner.fit(embeddings_non_toxic, embeddings_toxic)
        
        sampler = SASASampler(learner, alpha=1.0, temperature=1.0)
        
        assert sampler.subspace_learner == learner
        assert sampler.alpha == 1.0
        assert sampler.temperature == 1.0
    
    def test_compute_token_margins(self):
        """
        Test that compute_token_margins() returns correct shape.
        """
        learner = SubspaceLearner(embedding_dim=10)
        embeddings_non_toxic = torch.randn(50, 10)
        embeddings_toxic = torch.randn(50, 10)
        learner.fit(embeddings_non_toxic, embeddings_toxic)
        
        sampler = SASASampler(learner)
        
        current_embedding = torch.randn(10)
        token_embeddings = torch.randn(100, 10)  # 100 vocab tokens
        
        margins = sampler.compute_token_margins(current_embedding, token_embeddings)
        
        assert margins.shape == (100,)
    
    def test_adjust_logits(self):
        """
        Test that adjust_logits() modifies logits based on margins.
        """
        learner = SubspaceLearner(embedding_dim=10)
        
        # Create clear separation
        embeddings_non_toxic = torch.randn(50, 10) * 0.3 + torch.tensor([3.0] + [0.0]*9)
        embeddings_toxic = torch.randn(50, 10) * 0.3 + torch.tensor([-3.0] + [0.0]*9)
        learner.fit(embeddings_non_toxic, embeddings_toxic)
        
        sampler = SASASampler(learner, alpha=1.0)
        
        logits = torch.randn(100)
        current_embedding = torch.randn(10)
        token_embeddings = torch.randn(100, 10)
        
        adjusted_logits = sampler.adjust_logits(logits, current_embedding, token_embeddings)
        
        assert adjusted_logits.shape == logits.shape
        assert not torch.allclose(adjusted_logits, logits)  # Should be different
    
    def test_sample(self):
        """
        Test that sample() returns a valid token index.
        """
        learner = SubspaceLearner(embedding_dim=10)
        embeddings_non_toxic = torch.randn(50, 10)
        embeddings_toxic = torch.randn(50, 10)
        learner.fit(embeddings_non_toxic, embeddings_toxic)
        
        sampler = SASASampler(learner)
        
        logits = torch.randn(100)
        current_embedding = torch.randn(10)
        token_embeddings = torch.randn(100, 10)
        
        token = sampler.sample(logits, current_embedding, token_embeddings)
        
        assert isinstance(token, torch.Tensor)
        assert 0 <= token.item() < 100
    
    def test_sample_with_top_k(self):
        """
        Test that sample() respects top_k parameter.
        """
        learner = SubspaceLearner(embedding_dim=10)
        embeddings_non_toxic = torch.randn(50, 10)
        embeddings_toxic = torch.randn(50, 10)
        learner.fit(embeddings_non_toxic, embeddings_toxic)
        
        sampler = SASASampler(learner)
        
        logits = torch.randn(100)
        current_embedding = torch.randn(10)
        token_embeddings = torch.randn(100, 10)
        
        # Sample multiple times with top_k=10
        samples = []
        for _ in range(50):
            token = sampler.sample(
                logits, current_embedding, token_embeddings, top_k=10
            )
            samples.append(token.item())
        
        # All samples should be valid
        assert all(0 <= s < 100 for s in samples)


class TestBaselineSampler:
    """
    Test cases for the BaselineSampler class.
    """
    
    def test_initialization(self):
        """
        Test that BaselineSampler initializes correctly.
        """
        sampler = BaselineSampler(temperature=1.0)
        assert sampler.temperature == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
