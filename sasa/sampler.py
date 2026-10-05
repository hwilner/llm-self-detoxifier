"""
SASA Sampling Module

This module implements the Self-disciplined Autoregressive Sampling (SASA) algorithm
for toxicity reduction in language model generation.

SASA adjusts the sampling distribution during autoregressive decoding by incorporating
margin information from the learned toxic/non-toxic subspace.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any, Tuple
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
        subspace_learner: Trained subspace learner for margin computation. May be
            a SubspaceLearner or a MultiLayerSubspaceLearner (issue #39).
        alpha: Weight for margin term (higher = stronger detoxification).
        temperature: Sampling temperature for diversity control.
        margin_top_k: If set, margins are computed only for the top-k tokens by
            logit; all other tokens receive zero margin.
        gate_threshold: If set, steering is applied only when the current
            context margin is below this threshold (issue #43).
    """
    
    def __init__(
        self,
        subspace_learner: SubspaceLearner,
        alpha: float = 1.0,
        temperature: float = 1.0,
        margin_top_k: Optional[int] = None,
        gate_threshold: Optional[float] = None
    ):
        """
        Initialize SASA sampler.
        
        Args:
            subspace_learner: Trained SubspaceLearner (or MultiLayerSubspaceLearner).
            alpha: Weight for margin-based steering (default: 1.0).
                Higher values increase detoxification strength.
            temperature: Sampling temperature (default: 1.0).
                Lower values make sampling more deterministic.
            margin_top_k: If set, only the top-k tokens by logit receive a
                margin adjustment; all other tokens get zero margin.
            gate_threshold: If set, only steer when the context margin is
                below this value. E.g. 0.0 steers only when the context is
                on the toxic side of the boundary; small positive values
                also steer in the boundary's vicinity.
        """
        self.subspace_learner = subspace_learner
        self.alpha = alpha
        self.temperature = temperature
        self.margin_top_k = margin_top_k
        self.gate_threshold = gate_threshold

    def _is_multilayer_ensemble(self) -> bool:
        """True when the learner is a MultiLayerSubspaceLearner in ensemble mode."""
        return (
            getattr(self.subspace_learner, "selection", None) == "ensemble"
            and hasattr(self.subspace_learner, "learners")
        )
        
    def compute_token_margins(
        self,
        current_embedding: torch.Tensor,
        token_embeddings: torch.Tensor,
        current_all_layers: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> torch.Tensor:
        """
        Compute margins for all candidate tokens.
        
        For each candidate token, we compute the margin of the context that would
        result from appending that token. In multi-layer ensemble mode the
        approximation is applied per layer and margins are averaged (the input
        token embedding is reused as the per-layer token delta approximation).
        
        Args:
            current_embedding: Current context embedding, shape (embedding_dim,).
            token_embeddings: Embeddings of all vocabulary tokens,
                shape (vocab_size, embedding_dim).
            current_all_layers: All-layer context embeddings (tuple of
                (embedding_dim,) tensors); required in ensemble mode.
                
        Returns:
            Margin values for each token, shape (vocab_size,).
        """
        if self._is_multilayer_ensemble():
            if current_all_layers is None:
                raise ValueError(
                    "MultiLayerSubspaceLearner in ensemble mode requires "
                    "all-layer hidden states (adapter include_all_layers=True)"
                )
            learner = self.subspace_learner
            per_layer_next = {
                l: (current_all_layers[i].unsqueeze(0) + token_embeddings) / 2
                for i, l in enumerate(learner.layers)
            }
            # margins per layer: (n_layers, vocab_size) -> mean over layers
            stacked = torch.stack([
                learner.learners[l].compute_margin(per_layer_next[l])
                for l in learner.layers
            ], dim=0)
            return stacked.mean(dim=0)

        # For simplicity, we approximate the next context embedding as
        # a combination of current embedding and token embedding
        # In practice, this would be the actual embedding after appending the token
        next_embeddings = (current_embedding.unsqueeze(0) + token_embeddings) / 2
        
        # Compute margin for each candidate
        margins = self.subspace_learner.compute_margin(next_embeddings)
        
        return margins
    
    def should_steer(self, current_embedding: torch.Tensor) -> bool:
        """Decide whether to apply margin steering for the current context.

        With no gate_threshold, steering is always on (original SASA behavior).
        With a gate, steering activates only when the context margin falls
        below the threshold — i.e. near or inside the toxic region.
        """
        if self.gate_threshold is None:
            return True
        context_margin = self.subspace_learner.compute_margin(current_embedding)
        return bool(context_margin.item() < self.gate_threshold)
    
    def adjust_logits(
        self,
        logits: torch.Tensor,
        current_embedding: torch.Tensor,
        token_embeddings: torch.Tensor,
        current_all_layers: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> torch.Tensor:
        """
        Adjust logits based on margin to toxic subspace.
        
        This implements the core SASA algorithm: combining the original logits
        with margin-based steering to guide generation away from toxic content.
        
        If self.margin_top_k is set, margins are computed only for the top-k
        tokens by logit; remaining tokens receive zero margin. If
        self.gate_threshold is set and the current context is safely
        non-toxic, logits are returned unchanged.
        
        Args:
            logits: Original model logits, shape (vocab_size,).
            current_embedding: Current context embedding, shape (embedding_dim,).
            token_embeddings: Token embeddings for all vocabulary,
                shape (vocab_size, embedding_dim).
            current_all_layers: All-layer context embeddings (ensemble mode).
                
        Returns:
            Adjusted logits, shape (vocab_size,).
        """
        if not self.should_steer(current_embedding):
            return logits

        vocab_size = logits.shape[-1]

        if self.margin_top_k is not None and self.margin_top_k < vocab_size:
            # Restrict margin computation to the top-k candidate tokens
            top_indices = torch.topk(logits, self.margin_top_k).indices
            margins = torch.zeros_like(logits)
            candidate_margins = self.compute_token_margins(
                current_embedding, token_embeddings[top_indices], current_all_layers
            )
            margins[top_indices] = candidate_margins
        else:
            margins = self.compute_token_margins(
                current_embedding, token_embeddings, current_all_layers
            )
        
        # Adjust logits: logits_adjusted = logits + alpha * margins
        adjusted_logits = logits + self.alpha * margins
        
        return adjusted_logits
    
    def sample(
        self,
        logits: torch.Tensor,
        current_embedding: torch.Tensor,
        token_embeddings: torch.Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        current_all_layers: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> torch.Tensor:
        """
        Sample next token using SASA algorithm.
        
        Args:
            logits: Original model logits, shape (vocab_size,).
            current_embedding: Current context embedding, shape (embedding_dim,).
            token_embeddings: Token embeddings, shape (vocab_size, embedding_dim).
            top_k: If specified, only sample from top k tokens.
            top_p: If specified, use nucleus sampling with this threshold.
            current_all_layers: All-layer context embeddings (ensemble mode).
                
        Returns:
            Sampled token index.
        """
        # Adjust logits based on margin
        adjusted_logits = self.adjust_logits(
            logits,
            current_embedding,
            token_embeddings,
            current_all_layers
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
        return_scores: bool = False,
        use_cache: bool = True,
        adapter = None
    ) -> Dict[str, Any]:
        """
        Generate text using SASA sampling.
        
        Args:
            model: Language model to use for generation. Ignored when
                `adapter` is provided (pass None or the model).
            tokenizer: Corresponding tokenizer.
            prompt: Input prompt text.
            max_length: Maximum number of tokens to generate.
            device: Device to run on (defaults to model's device).
            top_k: Optional top-k filtering.
            top_p: Optional nucleus sampling threshold.
            return_scores: If True, return toxicity scores at each step.
            use_cache: If True (default), use the model's KV cache so each
                step processes only the newest token. Falls back to full
                forward passes if the model does not return past_key_values.
            adapter: Optional BackboneAdapter (see sasa/adapters.py). When
                provided, all model access goes through the adapter, making
                SASA architecture-agnostic.
                
        Returns:
            Dictionary containing:
            - 'text': Generated text
            - 'tokens': List of generated token IDs
            - 'scores': (Optional) Toxicity scores at each step
        """
        if adapter is None:
            # Backwards-compatible path: wrap the HF model transparently.
            from .adapters import HFTransformerAdapter
            adapter = HFTransformerAdapter(model)

        if self._is_multilayer_ensemble() and hasattr(adapter, "include_all_layers"):
            adapter.include_all_layers = True

        if device is None and model is not None:
            device = next(model.parameters()).device
        
        if model is not None:
            model.eval()
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if device is not None:
            input_ids = input_ids.to(device)
        
        # Get token embeddings from model
        token_embeddings = adapter.token_embeddings()
        
        generated_tokens = []
        scores = [] if return_scores else None
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # With a KV cache, only the newest token is processed after
                # the first (prompt) forward pass.
                if use_cache and past_key_values is not None:
                    step_input = input_ids[:, -1:]
                else:
                    step_input = input_ids

                step = adapter.forward_step(
                    step_input,
                    past_key_values=past_key_values if use_cache else None,
                    use_cache=use_cache
                )
                logits = step.logits
                current_embedding = step.hidden_state

                if use_cache:
                    # Some backbones (e.g. certain RNN/SSM implementations) do
                    # not return a cache; fall back to full forward passes.
                    past_key_values = step.past_key_values
                
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
                    top_p=top_p,
                    current_all_layers=step.hidden_states
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
        top_p: Optional[float] = None,
        use_cache: bool = True
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
            use_cache: If True (default), use the model's KV cache so each
                step processes only the newest token.
                
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
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_length):
                if use_cache and past_key_values is not None:
                    step_input = input_ids[:, -1:]
                else:
                    step_input = input_ids

                outputs = model(
                    step_input,
                    past_key_values=past_key_values if use_cache else None,
                    use_cache=use_cache
                )
                logits = outputs.logits[0, -1, :] / self.temperature

                if use_cache:
                    past_key_values = getattr(outputs, "past_key_values", None)
                
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
