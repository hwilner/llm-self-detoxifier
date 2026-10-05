"""
Tests for the circuit breaker (issue #45): generation aborts when the
context margin stays on the toxic side of the boundary for k consecutive
steps — a decode-time refusal an abliterated model cannot "forget".

Uses a tiny randomly initialized GPT-2 built from a local config.
"""

import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from sasa import SubspaceLearner, SASASampler, HFTransformerAdapter


VOCAB_SIZE = 99
EMB_DIM = 32


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text, return_tensors="pt"):
        return torch.tensor([[(ord(c) % (VOCAB_SIZE - 1)) + 1 for c in text]])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB_SIZE, n_embd=EMB_DIM, n_layer=2, n_head=2,
        n_positions=128, bos_token_id=0, eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


@pytest.fixture
def learner():
    torch.manual_seed(1)
    learner = SubspaceLearner(embedding_dim=EMB_DIM)
    learner.fit(torch.randn(32, EMB_DIM), torch.randn(32, EMB_DIM) + 1.0)
    return learner


def force_margin(learner, value):
    """Deterministically pin every context margin (test stub)."""
    learner.compute_margin = lambda emb: (
        torch.tensor(value) if emb.dim() == 1 else torch.full((emb.shape[0],), value)
    )


def test_circuit_breaker_fires_after_k_toxic_steps(tiny_model, learner):
    force_margin(learner, -1.0)  # always on the toxic side
    sampler = SASASampler(learner, alpha=1.0, gate_threshold=0.0, circuit_breaker_k=3)
    out = sampler.generate(
        tiny_model, TinyTokenizer(), "hello", max_length=50,
        adapter=HFTransformerAdapter(tiny_model),
    )
    assert out.get("stopped_by_circuit_breaker") is True
    assert len(out["tokens"]) < 50


def test_circuit_breaker_silent_when_safe(tiny_model, learner):
    force_margin(learner, 1.0)  # always safe
    sampler = SASASampler(learner, alpha=1.0, circuit_breaker_k=3)
    out = sampler.generate(
        tiny_model, TinyTokenizer(), "hello", max_length=8,
        adapter=HFTransformerAdapter(tiny_model),
    )
    assert "stopped_by_circuit_breaker" not in out
    assert len(out["tokens"]) == 8


def test_circuit_breaker_disabled_by_default(tiny_model, learner):
    force_margin(learner, -1.0)
    sampler = SASASampler(learner, alpha=1.0)  # no circuit_breaker_k
    out = sampler.generate(
        tiny_model, TinyTokenizer(), "hello", max_length=8,
        adapter=HFTransformerAdapter(tiny_model),
    )
    assert "stopped_by_circuit_breaker" not in out
    assert len(out["tokens"]) == 8
