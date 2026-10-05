"""
Tests for MultiLayerSubspaceLearner and multi-layer adapter support (#39).

Uses a tiny randomly initialized GPT-2 built from a local config — no network
access or pretrained weights required.
"""

import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from sasa import (
    SubspaceLearner,
    MultiLayerSubspaceLearner,
    SASASampler,
    HFTransformerAdapter,
    extract_embeddings_multilayer,
)


VOCAB_SIZE = 99
EMB_DIM = 32
LAYERS = [1, 2, 3]  # GPT-2 hidden_states: 0=embeddings, 1..n=blocks


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text, return_tensors="pt"):
        return torch.tensor([[(ord(c) % (VOCAB_SIZE - 1)) + 1 for c in text]])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, return_tensors="pt"):
        class Batch:
            def __init__(self, ids):
                self.data = {"input_ids": ids}
            def to(self, device):
                return self.data
        return Batch(self.encode(text))


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB_SIZE, n_embd=EMB_DIM, n_layer=3, n_head=2,
        n_positions=128, bos_token_id=0, eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def synthetic_by_layer(shift_layer, n=64, dim=EMB_DIM):
    """Signal planted only in `shift_layer`; other layers are noise."""
    torch.manual_seed(0)
    shift = torch.tensor([1.5] + [0.0] * (dim - 1))
    nt = {l: torch.randn(n, dim) for l in LAYERS}
    t = {l: torch.randn(n, dim) + (shift if l == shift_layer else 0.0) for l in LAYERS}
    return nt, t


def test_fit_and_select_best_layer():
    nt, t = synthetic_by_layer(shift_layer=2)
    ml = MultiLayerSubspaceLearner(EMB_DIM, LAYERS, selection="best").fit(nt, t)
    v_nt, v_t = synthetic_by_layer(shift_layer=2, n=32)
    scores = ml.separability_scores(v_nt, v_t)
    assert set(scores.keys()) == set(LAYERS)
    assert ml.select_layer(v_nt, v_t) == 2
    assert scores[2] > scores[1] and scores[2] > scores[3]


def test_ensemble_margin_is_layer_mean():
    nt, t = synthetic_by_layer(shift_layer=2)
    ml = MultiLayerSubspaceLearner(EMB_DIM, LAYERS, selection="ensemble").fit(nt, t)
    x = torch.randn(5, EMB_DIM)
    ensemble = ml.compute_margin({l: x for l in LAYERS})
    manual = torch.stack([ml.learners[l].compute_margin(x) for l in LAYERS]).mean(dim=0)
    assert torch.allclose(ensemble, manual)


def test_save_load_round_trip(tmp_path):
    nt, t = synthetic_by_layer(shift_layer=2)
    ml = MultiLayerSubspaceLearner(EMB_DIM, LAYERS, selection="ensemble").fit(nt, t)
    x = torch.randn(5, EMB_DIM)
    before = ml.compute_margin({l: x for l in LAYERS})
    ml.save(str(tmp_path / "ml.pt"))
    ml2 = MultiLayerSubspaceLearner(EMB_DIM, LAYERS, selection="ensemble")
    ml2.load(str(tmp_path / "ml.pt"))
    assert torch.allclose(ml2.compute_margin({l: x for l in LAYERS}), before)


def test_ensemble_generation_end_to_end(tiny_model):
    nt, t = synthetic_by_layer(shift_layer=2)
    ml = MultiLayerSubspaceLearner(EMB_DIM, LAYERS, selection="ensemble").fit(nt, t)
    sampler = SASASampler(ml, alpha=1.0, margin_top_k=20)
    adapter = HFTransformerAdapter(tiny_model)
    out = sampler.generate(None, TinyTokenizer(), "hello", max_length=6, adapter=adapter)
    assert len(out["tokens"]) >= 1
    assert adapter.include_all_layers  # auto-enabled for ensemble mode


def test_best_mode_with_hidden_layer_adapter(tiny_model):
    nt, t = synthetic_by_layer(shift_layer=2)
    ml = MultiLayerSubspaceLearner(EMB_DIM, LAYERS, selection="best").fit(nt, t)
    v_nt, v_t = synthetic_by_layer(shift_layer=2, n=32)
    best = ml.select_layer(v_nt, v_t)
    sampler = SASASampler(ml, alpha=1.0)
    adapter = HFTransformerAdapter(tiny_model, hidden_layer=best)
    out = sampler.generate(None, TinyTokenizer(), "hello", max_length=6, adapter=adapter)
    assert len(out["tokens"]) >= 1


def test_extract_embeddings_multilayer(tiny_model):
    emb = extract_embeddings_multilayer(
        tiny_model, TinyTokenizer(), ["hello", "world"], torch.device("cpu"), layers=LAYERS
    )
    assert set(emb.keys()) == set(LAYERS)
    for l in LAYERS:
        assert emb[l].shape == (2, EMB_DIM)


def test_invalid_selection_raises():
    with pytest.raises(ValueError):
        MultiLayerSubspaceLearner(EMB_DIM, LAYERS, selection="median")
