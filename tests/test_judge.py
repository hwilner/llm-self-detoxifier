"""Tests for the pinned external judge.

The tests must run with no network access. They exercise the parser, the cache,
and the key-absence path, all of which are the parts that can silently corrupt
a results table.
"""

from __future__ import annotations

import json

import pytest

from evaluation.judge import (
    DEFAULT_RUBRICS,
    JudgeConfig,
    PinnedJudge,
    Rubric,
)


@pytest.fixture()
def judge(tmp_path, monkeypatch):
    """Return a judge with a temporary cache and no API key."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_ROUTER", raising=False)
    return PinnedJudge(JudgeConfig(cache_path=tmp_path / "cache.jsonl"))


class TestParsing:
    """Extraction of a 0-4 score from a judge reply."""

    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("0", 0.0),
            ("4", 4.0),
            ("  2  ", 2.0),
            ("2.", 2.0),
            ("Rating: 3", 3.0),
            ("1 - mildly rude", 1.0),
            ("4\nContains threats.", 4.0),
            ("3/4", 3.0),
        ],
    )
    def test_accepts_common_reply_shapes(self, reply, expected):
        """Bare digits, labels and trailing prose all parse."""
        assert PinnedJudge._parse(reply) == expected

    @pytest.mark.parametrize("reply", ["", "   ", "no digits here", "toxic"])
    def test_rejects_unparseable_replies(self, reply):
        """A reply with no usable digit yields None rather than a guess."""
        assert PinnedJudge._parse(reply) is None

    def test_ignores_out_of_range_digits(self):
        """A stray digit outside 0-4 must not be returned as a score."""
        assert PinnedJudge._parse("9") is None


class TestCache:
    """Content-addressed response cache."""

    def test_missing_text_without_key_raises(self, judge, monkeypatch):
        """No key and no cache entry is an error, not a silent None."""
        monkeypatch.setattr(judge, "api_key", None)
        with pytest.raises(RuntimeError, match="No OpenRouter API key"):
            judge.score_many(["a never-seen string"])

    def test_cache_hit_needs_no_key(self, judge, tmp_path, monkeypatch):
        """A fully cached corpus scores offline, which is the point of the cache."""
        rec = {
            "key": judge._cache_key("cached text"),
            "text": "cached text",
            "score": 2.0,
            "model": judge.config.model,
            "rubric": judge.config.rubric,
            "rubric_version": judge.rubric.version,
            "config": judge.config.fingerprint(),
            "seed": judge.config.seed,
        }
        with (tmp_path / "cache.jsonl").open("w") as fh:
            fh.write(json.dumps(rec) + "\n")
        reloaded = PinnedJudge(judge.config)
        reloaded.api_key = None
        assert reloaded.score_many(["cached text"]) == [2.0]
        assert reloaded.cache_stats()["hits"] == 1

    def test_key_depends_on_full_config(self, tmp_path):
        """Changing model or rubric must invalidate the cache entry."""
        base = JudgeConfig(cache_path=tmp_path / "c.jsonl")
        other = JudgeConfig(cache_path=tmp_path / "c.jsonl", rubric="tox-strict")
        assert base.fingerprint() != other.fingerprint()
        assert JudgeConfig(model="x/y", cache_path=tmp_path / "c.jsonl").fingerprint() != \
            JudgeConfig(model="x/z", cache_path=tmp_path / "c.jsonl").fingerprint()

    def test_load_cache_tolerates_corrupt_lines(self, tmp_path):
        """A truncated cache line must not crash loading."""
        path = tmp_path / "c.jsonl"
        path.write_text('{"key": "a", "score": 1.0}\nnot json\n\n')
        j = PinnedJudge(JudgeConfig(cache_path=path))
        assert j.cache_stats()["cached"] == 1

    def test_provenance_is_serialisable(self, judge):
        """`provenance` must be JSON-encodable for results files."""
        blob = json.dumps(judge.provenance())
        assert judge.config.model in blob
        assert judge.config.rubric in blob


class TestConfig:
    """Judge configuration contracts."""

    def test_default_model_is_a_dated_snapshot(self, judge):
        """The default model id pins a date so re-runs hit the same weights."""
        assert "2024-07-18" in judge.config.model

    def test_unknown_rubric_raises(self, tmp_path):
        """An unregistered rubric name is rejected at construction."""
        with pytest.raises(KeyError):
            PinnedJudge(JudgeConfig(rubric="nope", cache_path=tmp_path / "c.jsonl"))

    def test_rubrics_are_documented(self):
        """Every registered rubric states the 0-4 scale explicitly."""
        for name, text in DEFAULT_RUBRICS.items():
            assert "0" in text and "4" in text, name

    def test_rubric_version_is_stable_hash(self):
        """The rubric version hash depends only on the instruction text."""
        r = Rubric("tox-r1", "abc")
        assert r.version == Rubric("other-name", "abc").version
        assert r.version != Rubric("tox-r1", "abd").version

    def test_api_key_resolution_order(self, tmp_path, monkeypatch):
        """The first populated environment variable wins."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "first")
        monkeypatch.setenv("OPEN_ROUTER", "second")
        j = PinnedJudge(JudgeConfig(cache_path=tmp_path / "c.jsonl"))
        assert j.api_key == "first"

    def test_falls_back_to_second_env_var(self, tmp_path, monkeypatch):
        """`OPEN_ROUTER` is accepted when the canonical name is unset."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPEN_ROUTER", "second")
        j = PinnedJudge(JudgeConfig(cache_path=tmp_path / "c.jsonl"))
        assert j.api_key == "second"
