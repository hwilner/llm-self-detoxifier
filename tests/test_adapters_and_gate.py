"""
Tests for the BackboneAdapter layer and margin-gated adaptive alpha.

Uses a tiny randomly initialized GPT-2 built from a local config — no network
access or pretrained weights required.
"""

import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from sasa import (
    SubspaceLearner,
    SASASampler,
    BackboneAdapter,
    HFTransformerAdapter,
    AdapterOutput,
)


VOCAB_SIZE = 99
EMB_DIM = 32


class TinyTokenizer:
    """Minimal tokenizer over a small integer vocab, for offline tests."""
    eos_token_id = 0

    def encode(self, text, return_tensors="pt"):
        ids = [(ord(c) % (VOCAB_SIZE - 1)) + 1 for c in text]
        return torch.tensor([ids])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_embd=EMB_DIM,
        n_layer=2,
        n_head=2,
        n_positions=128,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


@pytest.fixture
def learner():
    torch.manual_seed(1)
    non_toxic = torch.randn(32, EMB_DIM)
    toxic = torch.randn(32, EMB_DIM) + 1.0
    learner = SubspaceLearner(embedding_dim=EMB_DIM)
    learner.fit(non_toxic, toxic)
    return learner


# ---------- BackboneAdapter tests ----------

def test_hf_adapter_satisfies_protocol(tiny_model):
    adapter = HFTransformerAdapter(tiny_model)
    assert isinstance(adapter, BackboneAdapter)
    assert adapter.supports_incremental()


def test_hf_adapter_shapes(tiny_model):
    adapter = HFTransformerAdapter(tiny_model)
    out = adapter.forward_step(torch.tensor([[5, 10, 15]]))
    assert isinstance(out, AdapterOutput)
    assert out.logits.shape == (VOCAB_SIZE,)
    assert out.hidden_state.shape == (EMB_DIM,)
    assert out.past_key_values is not None
    assert adapter.token_embeddings().shape == (VOCAB_SIZE, EMB_DIM)


def test_generate_through_adapter_matches_direct(tiny_model, learner):
    """Adapter-driven generation should match the legacy direct path."""
    tokenizer = TinyTokenizer()
    sampler = SASASampler(subspace_learner=learner, alpha=1.0)

    torch.manual_seed(42)
    direct = sampler.generate(tiny_model, tokenizer, "hello", max_length=8)

    torch.manual_seed(42)
    via_adapter = sampler.generate(
        None, tokenizer, "hello", max_length=8,
        adapter=HFTransformerAdapter(tiny_model),
    )

    assert direct["tokens"] == via_adapter["tokens"]


# ---------- Margin-gated alpha tests ----------

def test_gate_off_when_safe(tiny_model, learner):
    """Safely non-toxic contexts should be decoded unmodified."""
    sampler = SASASampler(subspace_learner=learner, alpha=1.0, gate_threshold=0.0)

    # An embedding deep in the non-toxic class region
    safe_embedding = learner.params.mu_1 + 10 * learner.params.w_v
    logits = torch.randn(VOCAB_SIZE)
    token_embeddings = tiny_model.get_input_embeddings().weight

    adjusted = sampler.adjust_logits(logits, safe_embedding, token_embeddings)
    assert torch.equal(adjusted, logits)
    assert not sampler.should_steer(safe_embedding)


def test_gate_on_when_toxic(tiny_model, learner):
    """Toxic-side contexts should be steered."""
    sampler = SASASampler(subspace_learner=learner, alpha=1.0, gate_threshold=0.0)

    # An embedding deep in the toxic class region
    toxic_embedding = learner.params.mu_2 - 10 * learner.params.w_v
    torch.manual_seed(0)
    logits = torch.randn(VOCAB_SIZE)
    token_embeddings = tiny_model.get_input_embeddings().weight

    adjusted = sampler.adjust_logits(logits, toxic_embedding, token_embeddings)
    assert not torch.allclose(adjusted, logits)
    assert sampler.should_steer(toxic_embedding)


def test_no_gate_matches_original(tiny_model, learner):
    """gate_threshold=None must reproduce the original always-steer behavior."""
    gated = SASASampler(subspace_learner=learner, alpha=1.0, gate_threshold=None)
    plain = SASASampler(subspace_learner=learner, alpha=1.0)

    torch.manual_seed(0)
    logits = torch.randn(VOCAB_SIZE)
    emb = torch.randn(EMB_DIM)
    token_embeddings = tiny_model.get_input_embeddings().weight

    assert torch.allclose(
        gated.adjust_logits(logits, emb, token_embeddings),
        plain.adjust_logits(logits, emb, token_embeddings),
    )
