"""
SASA Sampling Module

This module implements the Self-disciplined Autoregressive Sampling (SASA) algorithm
for toxicity reduction in language model generation.

SASA adjusts the sampling distribution during autoregressive decoding by incorporating
margin information from the learned toxic/non-toxic subspace.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any
from .subspace_learner import SubspaceLearner


class SASASampler:
    """
    SASA sampler for controlled text generation.
    
    This sampler implements the constrained optimization approach from the SASA paper,
    which balances two objectives:
    1. Alignment: Maximize margin to stay away from toxic subspace
    2. Utility: Keep sampling distribution close to original (maintain fluency)
    
    The sampling distribution is: p = Softmax(alpha * m + logits)
    where m is the margin vector and alpha controls the strength of detoxification.
    
    Attributes:
        subspace_learner: Trained subspace learner for margin computation.
        alpha: Weight for margin term (higher = stronger detoxification).
        temperature: Sampling temperature for diversity control.
    """
    
    def __init__(
        self,
        subspace_learner: SubspaceLearner,
        alpha: float = 1.0,
        temperature: float = 1.0
    ):
        """
        Initialize SASA sampler.
        
        Args:
            subspace_learner: Trained SubspaceLearner instance.
            alpha: Weight for margin-based steering (default: 1.0).
                Higher values increase detoxification strength.
            temperature: Sampling temperature (default: 1.0).
                Lower values make sampling more deterministic.
        """
        self.subspace_learner = subspace_learner
        self.alpha = alpha
        self.temperature = temperature
        
    def compute_token_margins(
        self,
        current_embedding: torch.Tensor,
        token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute margins for all candidate tokens.
        
        For each candidate token, we compute the margin of the context that would
        result from appending that token.
        
        Args:
            current_embedding: Current context embedding, shape (embedding_dim,).
            token_embeddings: Embeddings of all vocabulary tokens,
                shape (vocab_size, embedding_dim).
                
        Returns:
            Margin values for each token, shape (vocab_size,).
        """
        vocab_size = token_embeddings.shape[0]
        
        # For simplicity, we approximate the next context embedding as
        # a combination of current embedding and token embedding
        # In practice, this would be the actual embedding after appending the token
        next_embeddings = (current_embedding.unsqueeze(0) + token_embeddings) / 2
        
        # Compute margin for each candidate
        margins = self.subspace_learner.compute_margin(next_embeddings)
        
        return margins
    
    def adjust_logits(
        self,
        logits: torch.Tensor,
        current_embedding: torch.Tensor,
        token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Adjust logits based on margin to toxic subspace.
        
        This implements the core SASA algorithm: combining the original logits
        with margin-based steering to guide generation away from toxic content.
        
        Args:
            logits: Original model logits, shape (vocab_size,).
            current_embedding: Current context embedding, shape (embedding_dim,).
            token_embeddings: Token embeddings for all vocabulary,
                shape (vocab_size, embedding_dim).
                
        Returns:
            Adjusted logits, shape (vocab_size,).
        """
        # Compute margins for all tokens
        margins = self.compute_token_margins(current_embedding, token_embeddings)
        
        # Adjust logits: logits_adjusted = logits + alpha * margins
        adjusted_logits = logits + self.alpha * margins
        
        return adjusted_logits
    
    def sample(
        self,
        logits: torch.Tensor,
        current_embedding: torch.Tensor,
        token_embeddings: torch.Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Sample next token using SASA algorithm.
        
        Args:
            logits: Original model logits, shape (vocab_size,).
            current_embedding: Current context embedding, shape (embedding_dim,).
            token_embeddings: Token embeddings, shape (vocab_size, embedding_dim).
            top_k: If specified, only sample from top k tokens.
            top_p: If specified, use nucleus sampling with this threshold.
                
        Returns:
            Sampled token index.
        """
        # Adjust logits based on margin
        adjusted_logits = self.adjust_logits(
            logits,
            current_embedding,
            token_embeddings
        )
        
        # Apply temperature
        adjusted_logits = adjusted_logits / self.temperature
        
        # Apply top-k filtering if specified
        if top_k is not None:
            indices_to_remove = adjusted_logits < torch.topk(adjusted_logits, top_k)[0][..., -1, None]
            adjusted_logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering if specified
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(adjusted_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            adjusted_logits[indices_to_remove] = float('-inf')
        
        # Sample from adjusted distribution
        probs = F.softmax(adjusted_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_length: int = 50,
        device: torch.device = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using SASA sampling.
        
        Args:
            model: Language model to use for generation.
            tokenizer: Corresponding tokenizer.
            prompt: Input prompt text.
            max_length: Maximum number of tokens to generate.
            device: Device to run on (defaults to model's device).
            top_k: Optional top-k filtering.
            top_p: Optional nucleus sampling threshold.
            return_scores: If True, return toxicity scores at each step.
                
        Returns:
            Dictionary containing:
            - 'text': Generated text
            - 'tokens': List of generated token IDs
            - 'scores': (Optional) Toxicity scores at each step
        """
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Get token embeddings from model
        token_embeddings = model.get_input_embeddings().weight
        
        generated_tokens = []
        scores = [] if return_scores else None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model outputs
                outputs = model(input_ids, output_hidden_states=True)
                logits = outputs.logits[0, -1, :]
                
                # Get current context embedding (last token's hidden state)
                current_embedding = outputs.hidden_states[-1][0, -1, :]
                
                # Compute toxicity score if requested
                if return_scores:
                    score = self.subspace_learner.classify(current_embedding).item()
                    scores.append(score)
                
                # Sample next token using SASA
                next_token = self.sample(
                    logits,
                    current_embedding,
                    token_embeddings,
                    top_k=top_k,
                    top_p=top_p
                )
                
                generated_tokens.append(next_token.item())
                
                # Append to input for next iteration
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        result = {
            'text': generated_text,
            'tokens': generated_tokens
        }
        
        if return_scores:
            result['scores'] = scores
        
        return result


class BaselineSampler:
    """
    Baseline sampler without SASA (standard autoregressive sampling).
    
    This sampler is used for comparison to demonstrate the effectiveness of SASA.
    
    Attributes:
        temperature: Sampling temperature.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize baseline sampler.
        
        Args:
            temperature: Sampling temperature (default: 1.0).
        """
        self.temperature = temperature
    
    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_length: int = 50,
        device: torch.device = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate text using standard sampling (no SASA).
        
        Args:
            model: Language model to use for generation.
            tokenizer: Corresponding tokenizer.
            prompt: Input prompt text.
            max_length: Maximum number of tokens to generate.
            device: Device to run on (defaults to model's device).
            top_k: Optional top-k filtering.
            top_p: Optional nucleus sampling threshold.
                
        Returns:
            Dictionary containing:
            - 'text': Generated text
            - 'tokens': List of generated token IDs
        """
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model outputs
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :] / self.temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering if specified
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        0, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_tokens.append(next_token.item())
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            'text': generated_text,
            'tokens': generated_tokens
        }
