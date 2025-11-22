"""
Subspace Learning Module for SASA

This module implements the subspace learning component of SASA (Self-disciplined
Autoregressive Sampling) as described in "Large Language Models can be Strong
Self-Detoxifiers" (arXiv:2410.03818).

The subspace learner builds a linear classifier that separates toxic and non-toxic
content in the embedding space using class-conditional Gaussian distributions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SubspaceParams:
    """
    Parameters defining the learned toxic/non-toxic subspace.
    
    Attributes:
        w_v: Weight vector for the linear classifier.
        b_v: Bias term for the linear classifier.
        mu_1: Mean vector for non-toxic class.
        mu_2: Mean vector for toxic class.
        sigma: Shared covariance matrix.
        embedding_dim: Dimension of the embedding space.
    """
    w_v: torch.Tensor
    b_v: torch.Tensor
    mu_1: torch.Tensor
    mu_2: torch.Tensor
    sigma: torch.Tensor
    embedding_dim: int


class SubspaceLearner:
    """
    Learns linear subspaces characterizing toxic vs non-toxic output.
    
    This class implements the subspace learning algorithm from the SASA paper,
    which uses class-conditional Gaussian distributions to model the embedding
    space and derives an optimal Bayes classifier in analytical form.
    
    The classifier has the form: f_v(c,x) = sign(w_v^T(g(c⊕x) - b_v))
    where g(c⊕x) is the context embedding.
    
    Attributes:
        embedding_dim: Dimension of the embedding space.
        params: Learned subspace parameters (None until fit() is called).
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize the subspace learner.
        
        Args:
            embedding_dim: Dimension of the embedding space from the LLM.
        """
        self.embedding_dim = embedding_dim
        self.params: Optional[SubspaceParams] = None
        
    def fit(
        self,
        embeddings_non_toxic: torch.Tensor,
        embeddings_toxic: torch.Tensor
    ) -> SubspaceParams:
        """
        Learn the toxic/non-toxic subspace from labeled embeddings.
        
        This method estimates class-conditional Gaussian parameters and derives
        the optimal Bayes classifier in closed form.
        
        Args:
            embeddings_non_toxic: Embeddings of non-toxic examples,
                shape (N1, embedding_dim).
            embeddings_toxic: Embeddings of toxic examples,
                shape (N2, embedding_dim).
                
        Returns:
            SubspaceParams object containing the learned parameters.
            
        Raises:
            ValueError: If embeddings have incorrect dimensions or insufficient samples.
        """
        if embeddings_non_toxic.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Non-toxic embeddings dimension {embeddings_non_toxic.shape[1]} "
                f"does not match expected dimension {self.embedding_dim}"
            )
        if embeddings_toxic.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Toxic embeddings dimension {embeddings_toxic.shape[1]} "
                f"does not match expected dimension {self.embedding_dim}"
            )
        if embeddings_non_toxic.shape[0] < 2 or embeddings_toxic.shape[0] < 2:
            raise ValueError("Need at least 2 samples per class for covariance estimation")
        
        N1 = embeddings_non_toxic.shape[0]
        N2 = embeddings_toxic.shape[0]
        
        # Compute class means
        mu_1 = torch.mean(embeddings_non_toxic, dim=0)
        mu_2 = torch.mean(embeddings_toxic, dim=0)
        
        # Compute shared covariance matrix
        centered_1 = embeddings_non_toxic - mu_1.unsqueeze(0)
        centered_2 = embeddings_toxic - mu_2.unsqueeze(0)
        
        cov_1 = torch.matmul(centered_1.T, centered_1)
        cov_2 = torch.matmul(centered_2.T, centered_2)
        
        sigma = (cov_1 + cov_2) / (N1 + N2 - 2)
        
        # Add regularization for numerical stability
        sigma = sigma + 1e-6 * torch.eye(self.embedding_dim, device=sigma.device)
        
        # Compute optimal classifier parameters
        # w_v = Sigma^{-1}(mu_1 - mu_2)
        # b_v = (mu_1 + mu_2) / 2
        try:
            sigma_inv = torch.linalg.inv(sigma)
        except RuntimeError:
            # If inversion fails, use pseudo-inverse
            sigma_inv = torch.linalg.pinv(sigma)
        
        w_v = torch.matmul(sigma_inv, (mu_1 - mu_2))
        b_v = (mu_1 + mu_2) / 2
        
        self.params = SubspaceParams(
            w_v=w_v,
            b_v=b_v,
            mu_1=mu_1,
            mu_2=mu_2,
            sigma=sigma,
            embedding_dim=self.embedding_dim
        )
        
        return self.params
    
    def classify(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Classify an embedding as toxic or non-toxic.
        
        Args:
            embedding: Context embedding to classify, shape (embedding_dim,)
                or (batch_size, embedding_dim).
                
        Returns:
            Classification score where positive values indicate non-toxic
            and negative values indicate toxic. Shape matches input.
            
        Raises:
            RuntimeError: If fit() has not been called yet.
        """
        if self.params is None:
            raise RuntimeError("Must call fit() before classify()")
        
        # Handle both single embedding and batch
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # f_v(c,x) = w_v^T(g(c⊕x) - b_v)
        centered = embedding - self.params.b_v.unsqueeze(0)
        scores = torch.matmul(centered, self.params.w_v)
        
        if squeeze_output:
            scores = scores.squeeze(0)
        
        return scores
    
    def compute_margin(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute the margin from an embedding to the toxic subspace.
        
        The margin is the signed distance to the decision boundary.
        Positive margin means the embedding is in the non-toxic region.
        
        Args:
            embedding: Context embedding, shape (embedding_dim,) or
                (batch_size, embedding_dim).
                
        Returns:
            Margin value(s). Positive = non-toxic, negative = toxic.
            
        Raises:
            RuntimeError: If fit() has not been called yet.
        """
        if self.params is None:
            raise RuntimeError("Must call fit() before compute_margin()")
        
        # Margin is the classification score normalized by ||w_v||
        scores = self.classify(embedding)
        w_norm = torch.norm(self.params.w_v)
        
        return scores / w_norm
    
    def save(self, path: str) -> None:
        """
        Save learned parameters to disk.
        
        Args:
            path: Path to save the parameters.
            
        Raises:
            RuntimeError: If fit() has not been called yet.
        """
        if self.params is None:
            raise RuntimeError("Must call fit() before save()")
        
        torch.save({
            'w_v': self.params.w_v,
            'b_v': self.params.b_v,
            'mu_1': self.params.mu_1,
            'mu_2': self.params.mu_2,
            'sigma': self.params.sigma,
            'embedding_dim': self.params.embedding_dim
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load learned parameters from disk.
        
        Args:
            path: Path to load the parameters from.
        """
        checkpoint = torch.load(path)
        
        self.embedding_dim = checkpoint['embedding_dim']
        self.params = SubspaceParams(
            w_v=checkpoint['w_v'],
            b_v=checkpoint['b_v'],
            mu_1=checkpoint['mu_1'],
            mu_2=checkpoint['mu_2'],
            sigma=checkpoint['sigma'],
            embedding_dim=checkpoint['embedding_dim']
        )


def extract_embeddings_from_model(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device
) -> torch.Tensor:
    """
    Extract context embeddings from a language model for given texts.
    
    This function processes texts through the model and extracts the embedding
    of the last token, which serves as the context embedding.
    
    Args:
        model: The language model (e.g., GPT-2, Llama).
        tokenizer: Corresponding tokenizer.
        texts: List of text strings to process.
        device: Device to run the model on.
        
    Returns:
        Tensor of embeddings, shape (len(texts), embedding_dim).
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get the last layer's hidden state for the last token
            last_hidden_state = outputs.hidden_states[-1]
            last_token_embedding = last_hidden_state[0, -1, :]
            
            embeddings.append(last_token_embedding.cpu())
    
    return torch.stack(embeddings)
