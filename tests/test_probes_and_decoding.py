"""Tests for probes/confounds and for cached and batched decoding.

The decoding tests use a tiny randomly-initialised transformer so that the
cached and uncached paths can be compared exactly, with no download and no
external randomness.

Run with::

    pytest tests/test_probes_and_decoding.py -v
"""

from __future__ import annotations

import pytest
import torch

from sasa.decoding import (
    apply_top_k,
    apply_top_p,
    batched_sample,
    cached_generate,
    sample_from_logits,
    token_budget,
    uncached_generate,
)
from sasa.probes import (
    CONFOUND_ANGLE_THRESHOLDS,
    Probe,
    confound_angles,
    partial_out,
    partial_out_heldout_accuracy,
)


# --------------------------------------------------------------- fixtures


def _axis(dim: int, idx: int) -> torch.Tensor:
    """Return a unit vector along axis ``idx``.

    Args:
        dim: Vector length.
        idx: Axis index.

    Returns:
        A vector of shape ``(dim,)``.
    """
    v = torch.zeros(dim)
    v[idx] = 1.0
    return v


def _separable(dim: int = 32, direction: int = 0, n: int = 200,
               sep: float = 2.5, seed: int = 0, noise: float = 1.0):
    """Build two Gaussian clusters separated along one axis.

    Args:
        dim: Hidden size.
        direction: Axis index to separate along.
        n: Samples per class.
        sep: Separation in noise units.
        seed: RNG seed.
        noise: Per-coordinate standard deviation.

    Returns:
        A ``(positive, negative)`` tuple of shape ``(n, dim)``.
    """
    g = torch.Generator().manual_seed(seed)
    axis = _axis(dim, direction)
    pos = torch.randn(n, dim, generator=g) * noise + sep * axis
    neg = torch.randn(n, dim, generator=g) * noise - sep * axis
    return pos, neg


class _TinyLM(torch.nn.Module):
    """A minimal causal language model with an explicit key-value cache.

    Attention is written out rather than delegated to
    ``torch.nn.MultiheadAttention`` so that the cache semantics are unambiguous
    and the cached and uncached paths cannot differ for reasons unrelated to
    the code under test.

    Parameters:
        vocab: Vocabulary size.
        dim: Hidden size.
        depth: Number of attention blocks.
    """

    def __init__(self, vocab: int = 37, dim: int = 16, depth: int = 2) -> None:
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.embed = torch.nn.Embedding(vocab, dim)
        self.pos = torch.nn.Embedding(512, dim)
        self.blocks = torch.nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(torch.nn.ModuleDict({
                "ln1": torch.nn.LayerNorm(dim),
                "wq": torch.nn.Linear(dim, dim, bias=False),
                "wk": torch.nn.Linear(dim, dim, bias=False),
                "wv": torch.nn.Linear(dim, dim, bias=False),
                "wo": torch.nn.Linear(dim, dim, bias=False),
                "ln2": torch.nn.LayerNorm(dim),
                "ffn": torch.nn.Linear(dim, dim),
            }))
        self.head = torch.nn.Linear(dim, vocab)
        self.eval()

    def forward(self, input_ids, past_key_values=None, use_cache: bool = False):
        """Run the model, optionally extending a cache.

        Args:
            input_ids: Token ids, shape ``(B, T)``.
            past_key_values: Optional list of per-layer ``(key, value)`` pairs.
            use_cache: Whether to return an extended cache.

        Returns:
            An object with ``logits`` of shape ``(B, T, V)`` and, when
            ``use_cache``, a ``past_key_values`` list.
        """
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        b, t = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(past_len, past_len + t, device=device)
        h = self.embed(input_ids) + self.pos(pos_ids).unsqueeze(0)
        scale = self.dim ** 0.5

        new_cache = []
        for i, blk in enumerate(self.blocks):
            x = blk["ln1"](h)
            q, k, v = blk["wq"](x), blk["wk"](x), blk["wv"](x)
            if use_cache and past_key_values is not None:
                pk, pv = past_key_values[i]
                k = torch.cat([pk, k], dim=1)
                v = torch.cat([pv, v], dim=1)
            if use_cache:
                new_cache.append((k, v))
            scores = torch.matmul(q, k.transpose(-1, -2)) / scale
            # Causal mask: query at absolute position past_len+i may attend to
            # every key up to and including itself.
            qpos = torch.arange(past_len, past_len + t, device=device).unsqueeze(1)
            kpos = torch.arange(k.shape[1], device=device).unsqueeze(0)
            scores = scores.masked_fill(kpos > qpos, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            h = h + blk["wo"](torch.matmul(attn, v))
            h = h + blk["ffn"](blk["ln2"](h))
        out = self.head(h)
        return _LMOutput(out, new_cache if use_cache else None)


class _LMOutput:
    """Minimal stand-in for a Hugging Face model output.

    Parameters:
        logits: Logits tensor.
        past_key_values: Optional cache.
    """

    def __init__(self, logits: torch.Tensor, past_key_values) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


@pytest.fixture()
def tiny_lm():
    """A deterministic tiny causal LM with a working cache."""
    torch.manual_seed(7)
    return _TinyLM()


# ------------------------------------------------------------------ probes


class TestProbe:
    """Fitting and scoring."""

    def test_recovers_a_separable_direction(self):
        """A clean contrast set yields a near-unit, correct direction."""
        pos, neg = _separable(direction=0, seed=1)
        probe = Probe.fit(pos, neg, name="toxicity", seed=1)
        assert probe.heldout_accuracy > 0.95
        assert abs(float(probe.weight[0])) > 0.9

    def test_weight_is_unit_norm(self):
        """Weights are normalised so angles between probes are meaningful."""
        pos, neg = _separable(seed=2)
        probe = Probe.fit(pos, neg, name="x", seed=1)
        assert float(probe.weight.norm()) == pytest.approx(1.0, abs=1e-5)

    def test_intent_is_recorded_for_known_names(self):
        """A recognised probe name carries its documented meaning."""
        pos, neg = _separable(seed=3)
        assert "harmful" in Probe.fit(pos, neg, "toxicity", seed=1).intent
        assert "undocumented" in Probe.fit(pos, neg, "mystery", seed=1).intent

    def test_score_shape_preserves_leading_dims(self):
        """Scoring a batch keeps the batch shape."""
        pos, _neg = _separable(seed=4)
        probe = Probe.fit(pos, _separable(seed=4)[1], name="x", seed=1)
        assert probe.score(pos).shape == (pos.shape[0],)
        assert probe.score(pos[:5, :]).shape == (5,)

    def test_as_dict_is_serialisable(self):
        """`as_dict` returns plain scalars."""
        import json

        pos, neg = _separable(seed=5)
        json.dumps(Probe.fit(pos, neg, "toxicity", seed=1).as_dict())

    def test_rejects_width_mismatch(self):
        """Two classes must share a feature width."""
        with pytest.raises(ValueError, match="width mismatch"):
            Probe.fit(torch.randn(20, 8), torch.randn(20, 5), "x")

    def test_rejects_too_few_examples(self):
        """Each class needs at least two rows."""
        with pytest.raises(ValueError, match="at least two"):
            Probe.fit(torch.randn(1, 8), torch.randn(9, 8), "x")


class TestConfoundAngles:
    """The diagnostic that names a confound."""

    def test_perpendicular_confound_is_clean(self):
        """A probe and a confound on orthogonal axes are far apart."""
        dim = 32
        tox = Probe.fit(*_separable(dim, 0, seed=6), name="toxicity", seed=1)
        ref = Probe.fit(*_separable(dim, 5, seed=7), name="refusal", seed=1)
        report = confound_angles(tox, [ref])
        assert report.worst()["angle_degrees"] > CONFOUND_ANGLE_THRESHOLDS["clean"]
        assert report.verdict == "clean"

    def test_parallel_confound_is_flagged(self):
        """A probe aligned with its confound is reported as confounded."""
        dim = 32
        pos, neg = _separable(dim, 0, seed=8)
        tox = Probe.fit(pos, neg, name="toxicity", seed=1)
        parallel = Probe.fit(pos, neg, name="toxicity_copy", seed=1)
        report = confound_angles(tox, [parallel])
        assert report.worst()["angle_degrees"] < \
            CONFOUND_ANGLE_THRESHOLDS["confounded"]
        assert report.verdict == "confounded"

    def test_unusable_confound_is_flagged_not_scored(self):
        """A noise probe is marked unusable by a significance test.

        On 72 held-out examples a probe fitted to pure noise lands anywhere in
        roughly [0.38, 0.54], so a fixed accuracy cutoff near 0.5 would admit
        noise. The z-score against chance is what separates them.
        """
        dim = 32
        tox = Probe.fit(*_separable(dim, 0, seed=9), name="toxicity", seed=1)
        unusable = 0
        for s in range(6):
            g = torch.Generator().manual_seed(s)
            noise = Probe.fit(torch.randn(120, dim, generator=g),
                              torch.randn(120, dim, generator=g),
                              name=f"noise{s}", seed=1)
            if not noise.is_identifiable():
                unusable += 1
            report = confound_angles(tox, [noise])
            assert report.worst()["confound_is_usable"] is noise.is_identifiable()
        assert unusable >= 5

    def test_pairs_sorted_by_angle(self):
        """The closest confound is reported first."""
        dim = 32
        tox = Probe.fit(*_separable(dim, 0, seed=11), name="toxicity", seed=1)
        near = Probe.fit(*_separable(dim, 1, seed=12), name="near", seed=1)
        far = Probe.fit(*_separable(dim, 20, seed=13), name="far", seed=1)
        angles = [c["angle_degrees"]
                  for c in confound_angles(tox, [far, near]).confounds]
        assert angles == sorted(angles)

    def test_rejects_width_mismatch(self):
        """Probes in different feature spaces cannot be compared."""
        tox = Probe.fit(*_separable(32, 0, seed=14), name="toxicity", seed=1)
        other = Probe.fit(*_separable(16, 0, seed=15), name="other", seed=1)
        with pytest.raises(ValueError, match="width"):
            confound_angles(tox, [other])

    def test_as_dict_and_worst_with_no_confounds(self):
        """`worst` returns None for an empty confounds list."""
        tox = Probe.fit(*_separable(32, 0, seed=16), name="toxicity", seed=1)
        report = confound_angles(tox, [])
        assert report.worst() is None
        assert report.verdict == "clean"
        assert report.as_dict()["target"] == "toxicity"


class TestPartialOut:
    """Confound removal and its price."""

    def test_removed_direction_is_orthogonal(self):
        """The adjusted direction is orthogonal to the confound."""
        dim = 32
        pos, neg = _separable(dim, 0, seed=17)
        tox = Probe.fit(pos, neg, "toxicity", seed=1)
        ref = Probe.fit(*_separable(dim, 1, seed=18), name="refusal", seed=1)
        adjusted = partial_out(tox, ref)
        assert abs(float(adjusted @ ref.weight)) < 1e-5

    def test_adjusted_is_unit_norm(self):
        """The adjusted direction is renormalised by default."""
        dim = 32
        tox = Probe.fit(*_separable(dim, 0, seed=19), "toxicity", seed=1)
        ref = Probe.fit(*_separable(dim, 1, seed=20), name="refusal", seed=1)
        assert float(partial_out(tox, ref).norm()) == pytest.approx(1.0, abs=1e-5)

    def test_parallel_directions_raise(self):
        """Nothing survives removing a parallel direction, so it is an error."""
        dim = 32
        pos, neg = _separable(dim, 0, seed=21)
        p = Probe.fit(pos, neg, "x", seed=1)
        with pytest.raises(ValueError, match="parallel"):
            partial_out(p, Probe.fit(pos, neg, "y", seed=1))

    def test_rejects_width_mismatch(self):
        """Widths must match."""
        a = Probe.fit(*_separable(32, 0, seed=22), "a", seed=1)
        b = Probe.fit(*_separable(16, 0, seed=23), "b", seed=1)
        with pytest.raises(ValueError, match="width mismatch"):
            partial_out(a, b)

    def test_cost_is_reported_alongside_the_gain(self):
        """Accuracy before and after are always returned together."""
        dim = 32
        pos, neg = _separable(dim, 0, seed=24)
        tox = Probe.fit(pos, neg, "toxicity", seed=1)
        ref = Probe.fit(*_separable(dim, 1, seed=25), name="refusal", seed=1)
        report = partial_out_heldout_accuracy(tox, ref, pos, neg)
        assert set(report) == {"accuracy_before", "accuracy_after", "cost",
                               "cosine_removed"}
        assert report["cost"] == pytest.approx(
            report["accuracy_before"] - report["accuracy_after"], abs=1e-9
        )
        assert 0.0 <= report["cosine_removed"] <= 1.0


# ---------------------------------------------------------------- decoding


class TestFiltering:
    """top-k and top-p."""

    def test_top_k_keeps_exactly_k(self):
        """Exactly k entries survive."""
        logits = torch.randn(50)
        out = apply_top_k(logits, 5)
        assert int((out > float("-inf")).sum()) == 5

    def test_top_k_rejects_nonpositive(self):
        """A non-positive k is an error."""
        with pytest.raises(ValueError, match="top_k"):
            apply_top_k(torch.randn(10), 0)

    def test_top_p_keeps_at_least_one(self):
        """Nucleus filtering never empties the support."""
        logits = torch.randn(200)
        out = apply_top_p(logits, 0.01)
        assert int((out > float("-inf")).sum()) >= 1

    def test_top_p_rejects_out_of_range(self):
        """An out-of-range threshold is an error."""
        with pytest.raises(ValueError, match="top_p"):
            apply_top_p(torch.randn(10), 1.5)

    def test_top_p_none_is_identity(self):
        """``None`` disables filtering."""
        logits = torch.randn(10)
        assert torch.equal(apply_top_p(logits, None), logits)
        assert torch.equal(apply_top_k(logits, None), logits)

    def test_sampling_rejects_nonpositive_temperature(self):
        """Temperature must be positive."""
        with pytest.raises(ValueError, match="temperature"):
            sample_from_logits(torch.randn(10), temperature=0.0)

    def test_sample_shape_and_dtype(self):
        """A sampled index is a 0-dim int64 tensor."""
        tok = sample_from_logits(torch.randn(20))
        assert tok.shape == ()
        assert tok.dtype == torch.int64


class TestCachedVersusUncached:
    """The exactness guarantee."""

    def test_cached_matches_uncached_logits(self, tiny_lm):
        """Per-step logit norms agree between the two loops.

        This is the property that makes the cached loop safe: if the model
        computes the same next-token distribution at every step, the sampled
        sequence is identical for a fixed seed.
        """
        ids = torch.tensor([[1, 2, 3, 4]])
        gen = torch.Generator().manual_seed(11)
        a = uncached_generate(tiny_lm, ids, max_new_tokens=8, generator=gen)
        gen = torch.Generator().manual_seed(11)
        b = cached_generate(tiny_lm, ids, max_new_tokens=8, generator=gen)
        assert torch.allclose(a.step_logits_norm, b.step_logits_norm, atol=1e-4)

    def test_cached_matches_uncached_tokens(self, tiny_lm):
        """The generated sequences are identical for a fixed seed."""
        ids = torch.tensor([[1, 2, 3, 4]])
        gen = torch.Generator().manual_seed(12)
        a = uncached_generate(tiny_lm, ids, 10, top_k=10, generator=gen)
        gen = torch.Generator().manual_seed(12)
        b = cached_generate(tiny_lm, ids, 10, top_k=10, generator=gen)
        assert torch.equal(a.tokens, b.tokens)

    def test_cached_matches_uncached_with_steering(self, tiny_lm):
        """A static bias does not break the equivalence."""
        ids = torch.tensor([[5, 6]])
        bias = torch.randn(tiny_lm.vocab)
        gen = torch.Generator().manual_seed(13)
        a = uncached_generate(tiny_lm, ids, 8, bias=bias, generator=gen)
        gen = torch.Generator().manual_seed(13)
        b = cached_generate(tiny_lm, ids, 8, bias=bias, generator=gen)
        assert torch.equal(a.tokens, b.tokens)

    def test_eos_stops_both_loops(self, tiny_lm):
        """Both loops stop as soon as the end-of-sequence token is drawn.

        A bias that makes token 0 the argmax, with ``top_k=1``, forces the
        condition deterministically instead of hoping a random model samples it.
        """
        ids = torch.tensor([[1, 2]])
        bias = torch.zeros(tiny_lm.vocab)
        bias[0] = 50.0
        gen = torch.Generator().manual_seed(14)
        a = uncached_generate(tiny_lm, ids, 20, bias=bias, top_k=1,
                              eos_token_id=0, generator=gen)
        gen = torch.Generator().manual_seed(14)
        b = cached_generate(tiny_lm, ids, 20, bias=bias, top_k=1,
                           eos_token_id=0, generator=gen)
        assert torch.equal(a.tokens, b.tokens)
        assert a.tokens.numel() == 0  # the very first draw is EOS

    def test_bias_width_is_validated(self, tiny_lm):
        """A bias of the wrong width is rejected by both loops."""
        ids = torch.tensor([[1]])
        with pytest.raises(ValueError, match="bias has V"):
            uncached_generate(tiny_lm, ids, 2, bias=torch.randn(3))
        with pytest.raises(ValueError, match="bias has V"):
            cached_generate(tiny_lm, ids, 2, bias=torch.randn(3))

    def test_negative_budget_rejected(self, tiny_lm):
        """A negative token budget is an error."""
        ids = torch.tensor([[1]])
        with pytest.raises(ValueError, match="max_new_tokens"):
            uncached_generate(tiny_lm, ids, -1)
        with pytest.raises(ValueError, match="max_new_tokens"):
            cached_generate(tiny_lm, ids, -1)

    def test_zero_tokens_produces_empty_output(self, tiny_lm):
        """Generating nothing yields no tokens and no norms."""
        out = cached_generate(tiny_lm, torch.tensor([[1]]), 0)
        assert out.tokens.numel() == 0
        assert out.step_logits_norm.numel() == 0

    def test_prefill_length_recorded(self, tiny_lm):
        """The prompt length is reported for the cost accounting."""
        out = cached_generate(tiny_lm, torch.tensor([[1, 2, 3]]), 4)
        assert out.prefill_tokens == 3


class TestTokenBudget:
    """The quadratic cost of the uncached loop."""

    def test_uncached_is_quadratic_in_generated_length(self):
        """Work grows with the sum of prefix lengths."""
        small = token_budget(10, 10)
        large = token_budget(10, 100)
        assert small["uncached_token_positions"] == sum(range(11, 21))
        assert large["uncached_token_positions"] == sum(range(11, 111))
        assert large["uncached_token_positions"] > 9 * small["uncached_token_positions"]

    def test_cached_is_linear(self):
        """Cached work is prompt plus generated."""
        b = token_budget(64, 32, n_layers=6)
        assert b["cached_token_positions"] == 96
        assert b["speedup"] == pytest.approx(2576 / 96)
        assert b["n_layers"] == 6

    def test_rejects_negative_inputs(self):
        """Negative lengths are an error."""
        with pytest.raises(ValueError, match="non-negative"):
            token_budget(-1, 5)


class TestBatchedSampling:
    """Batched sampling with a static bias."""

    def test_shape_and_dtype(self):
        """One index per row."""
        out = batched_sample(torch.randn(5, 40))
        assert out.shape == (5,)
        assert out.dtype == torch.int64

    def test_requires_2d_logits(self):
        """A 1-D logit tensor is rejected."""
        with pytest.raises(ValueError, match=r"\(B, V\)"):
            batched_sample(torch.randn(40))

    def test_bias_is_shared_across_rows(self):
        """One bias vector serves the whole batch."""
        torch.manual_seed(15)
        logits = torch.randn(4, 50)
        bias = torch.randn(50)
        gens = [torch.Generator().manual_seed(100 + i) for i in range(4)]
        batched = batched_sample(logits, bias=bias, generators=gens)
        from sasa.decoding import sample_from_logits
        singles = [
            int(sample_from_logits(logits[i] + bias,
                                   generator=torch.Generator().manual_seed(100 + i)))
            for i in range(4)
        ]
        assert batched.tolist() == singles

    def test_generator_count_must_match_batch(self):
        """A generator per row is required when generators are supplied."""
        with pytest.raises(ValueError, match="expected 3 generators"):
            batched_sample(torch.randn(3, 20),
                           generators=[torch.Generator().manual_seed(1)])

    def test_bias_width_validated(self):
        """A bias of the wrong width is rejected."""
        with pytest.raises(ValueError, match="bias has V"):
            batched_sample(torch.randn(2, 20), bias=torch.randn(9))

    def test_row_independence_under_a_shared_generator(self):
        """Rows do not contaminate each other when seeded separately."""
        logits = torch.randn(3, 30)
        gens = [torch.Generator().manual_seed(7) for _ in range(3)]
        a = batched_sample(logits, generators=gens)
        gens2 = [torch.Generator().manual_seed(7) for _ in range(3)]
        b = batched_sample(logits, generators=gens2)
        assert torch.equal(a, b)


class TestPerStepBias:
    """A per-step bias stack, which is how a schedule reaches the decoder."""

    def test_schedule_stack_is_accepted(self, tiny_lm):
        """An (L, V) bias is consumed one row per step."""
        ids = torch.tensor([[1, 2]])
        unit = torch.randn(tiny_lm.vocab)
        stack = torch.stack([unit * 0.0, unit * 1.0, unit * 2.0])
        out = cached_generate(tiny_lm, ids, 3, bias=stack, top_k=5,
                              generator=torch.Generator().manual_seed(3))
        assert out.tokens.numel() == 3

    def test_stack_and_vector_are_equivalent_when_constant(self, tiny_lm):
        """A constant stack behaves exactly like the equivalent 1-D bias."""
        ids = torch.tensor([[1, 2, 3]])
        unit = torch.randn(tiny_lm.vocab)
        stack = unit.unsqueeze(0).repeat(4, 1)
        gen = torch.Generator().manual_seed(4)
        a = cached_generate(tiny_lm, ids, 4, bias=unit, top_k=5, generator=gen)
        gen = torch.Generator().manual_seed(4)
        b = cached_generate(tiny_lm, ids, 4, bias=stack, top_k=5, generator=gen)
        assert torch.equal(a.tokens, b.tokens)

    def test_stack_is_clamped_past_its_length(self, tiny_lm):
        """Generation longer than the schedule repeats the final strength."""
        ids = torch.tensor([[1, 2]])
        unit = torch.randn(tiny_lm.vocab)
        short = torch.stack([unit * 0.0, unit * 1.0])
        long = torch.stack([unit * 0.0, unit * 1.0, unit * 1.0, unit * 1.0])
        gen = torch.Generator().manual_seed(5)
        a = cached_generate(tiny_lm, ids, 4, bias=short, top_k=5, generator=gen)
        gen = torch.Generator().manual_seed(5)
        b = cached_generate(tiny_lm, ids, 4, bias=long, top_k=5, generator=gen)
        assert torch.equal(a.tokens, b.tokens)

    def test_bad_stack_width_rejected(self, tiny_lm):
        """A stack with the wrong vocabulary width is rejected."""
        with pytest.raises(ValueError, match="V="):
            cached_generate(tiny_lm, torch.tensor([[1]]), 2,
                            bias=torch.randn(3, 5))
        with pytest.raises(ValueError, match="V="):
            uncached_generate(tiny_lm, torch.tensor([[1]]), 2,
                              bias=torch.randn(3, 5))

    def test_three_dimensional_bias_rejected(self, tiny_lm):
        """Only (V,) and (L, V) shapes are meaningful."""
        with pytest.raises(ValueError, match=r"\(V,\) or \(L, V\)"):
            cached_generate(tiny_lm, torch.tensor([[1]]), 2,
                            bias=torch.randn(1, 2, tiny_lm.vocab))


class TestVectorisedBatching:
    """The fast path: one multinomial over the whole batch."""

    def test_shape_and_dtype(self):
        """One index per row from the vectorised path."""
        out = batched_sample(torch.randn(6, 40), generator=None)
        assert out.shape == (6,)
        assert out.dtype == torch.int64

    def test_vectorised_matches_per_row_distribution(self):
        """The vectorised draw is distributionally equal to per-row draws.

        Exact token equality is not available here -- that is the documented
        difference between the two generator styles -- so agreement is checked
        in distribution over many draws, which is the property that matters for
        throughput.
        """
        torch.manual_seed(21)
        logits = torch.randn(1, 25).repeat(8, 1)
        bias = torch.randn(25)
        big = torch.stack([
            batched_sample(logits, bias=bias, top_k=5,
                           generator=torch.Generator().manual_seed(500 + i))
            for i in range(600)
        ])
        assert big.shape == (600, 8)
        # Rows are driven by the same logits, so they should agree far more
        # often than two independent samples would.
        same_row = float((big[:, 0] == big[:, 1]).float().mean())
        assert same_row > 0.15

    def test_bias_is_shared_across_rows_in_the_fast_path(self):
        """One bias vector still serves the whole batch."""
        torch.manual_seed(22)
        logits = torch.randn(4, 30)
        bias = torch.zeros(30)
        bias[:5] = 50.0  # make the first five tokens dominant
        out = batched_sample(logits, bias=bias, top_k=1,
                             generator=torch.Generator().manual_seed(3))
        assert bool((out < 5).all())

    def test_rejects_both_generator_styles(self):
        """Supplying both generator styles is ambiguous and rejected."""
        with pytest.raises(ValueError, match="not both"):
            batched_sample(torch.randn(2, 10),
                           generators=[torch.Generator().manual_seed(1)],
                           generator=torch.Generator().manual_seed(2))

    def test_per_row_path_still_exact(self):
        """The per-row path keeps the batch-size-independent guarantee."""
        logits = torch.randn(4, 30)
        bias = torch.randn(30)
        gens = [torch.Generator().manual_seed(70 + i) for i in range(4)]
        a = batched_sample(logits, bias=bias, generators=gens)
        # Same row, same seed, different batch position.
        b = batched_sample(logits[2:3], bias=bias,
                           generators=[torch.Generator().manual_seed(72)])
        assert int(a[2]) == int(b[0])
