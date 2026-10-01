"""
Tests for KV-cache generation, top-k margin restriction, and the
SASA LogitsProcessor integration.

Uses a tiny randomly initialized GPT-2 built from a local config — no network
access or pretrained weights required.
"""

import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessorList

from sasa import SubspaceLearner, SASASampler, BaselineSampler, SASALogitsProcessor


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


@pytest.fixture
def tokenizer():
    return TinyTokenizer()


def test_kv_cache_parity(tiny_model, learner, tokenizer):
    """Generation with and without KV cache should produce identical tokens."""
    sampler = SASASampler(subspace_learner=learner, alpha=1.0)

    torch.manual_seed(42)
    out_cached = sampler.generate(
        tiny_model, tokenizer, "hello world", max_length=10, use_cache=True
    )
    torch.manual_seed(42)
    out_uncached = sampler.generate(
        tiny_model, tokenizer, "hello world", max_length=10, use_cache=False
    )

    assert out_cached["tokens"] == out_uncached["tokens"]


def test_kv_cache_parity_baseline(tiny_model, tokenizer):
    sampler = BaselineSampler()

    torch.manual_seed(42)
    out_cached = sampler.generate(
        tiny_model, tokenizer, "hello world", max_length=10, use_cache=True
    )
    torch.manual_seed(42)
    out_uncached = sampler.generate(
        tiny_model, tokenizer, "hello world", max_length=10, use_cache=False
    )

    assert out_cached["tokens"] == out_uncached["tokens"]


def test_margin_top_k_zero_tail(tiny_model, learner):
    """With margin_top_k set, only top-k tokens receive nonzero margin."""
    k = 10
    sampler = SASASampler(subspace_learner=learner, alpha=1.0, margin_top_k=k)

    torch.manual_seed(0)
    logits = torch.randn(VOCAB_SIZE)
    current_embedding = torch.randn(EMB_DIM)
    token_embeddings = tiny_model.get_input_embeddings().weight

    adjusted = sampler.adjust_logits(logits, current_embedding, token_embeddings)
    margins = adjusted - logits  # alpha = 1.0

    top_indices = torch.topk(logits, k).indices
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    mask[top_indices] = True

    assert torch.all(margins[~mask] == 0)
    assert torch.any(margins[mask] != 0)


def test_margin_top_k_none_matches_full(tiny_model, learner):
    """margin_top_k=None should match the original full-vocab behavior."""
    full = SASASampler(subspace_learner=learner, alpha=1.0)
    capped = SASASampler(subspace_learner=learner, alpha=1.0, margin_top_k=None)

    torch.manual_seed(0)
    logits = torch.randn(VOCAB_SIZE)
    current_embedding = torch.randn(EMB_DIM)
    token_embeddings = tiny_model.get_input_embeddings().weight

    assert torch.allclose(
        full.adjust_logits(logits, current_embedding, token_embeddings),
        capped.adjust_logits(logits, current_embedding, token_embeddings),
    )


def test_logits_processor_generate_smoke(tiny_model, learner):
    """SASALogitsProcessor should run inside model.generate without error."""
    processor = SASALogitsProcessor(learner, tiny_model, alpha=1.0, margin_top_k=20)

    input_ids = torch.tensor([[5, 10, 15]])
    out = tiny_model.generate(
        input_ids,
        max_new_tokens=8,
        do_sample=False,
        logits_processor=LogitsProcessorList([processor]),
    )
    processor.close()

    assert out.shape[1] > input_ids.shape[1]


def test_logits_processor_adjusts_scores(tiny_model, learner):
    """Processor output should differ from raw scores when margins are nonzero."""
    processor = SASALogitsProcessor(learner, tiny_model, alpha=10.0)

    # Fire the hook with a real forward pass
    input_ids = torch.tensor([[5, 10, 15]])
    with torch.no_grad():
        outputs = tiny_model(input_ids)
    scores = outputs.logits[:, -1, :]

    adjusted = processor(input_ids, scores)
    processor.close()

    assert not torch.allclose(adjusted, scores)
