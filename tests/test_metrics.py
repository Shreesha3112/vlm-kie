"""Tests for evaluation metrics."""

import pytest
from vlm_kie.eval.metrics import (
    aggregate_metrics,
    compute_field_metrics,
    exact_match,
    partial_match,
    token_f1,
)


class TestExactMatch:
    def test_identical(self):
        assert exact_match("INV-001", "INV-001") is True

    def test_case_insensitive(self):
        assert exact_match("INV-001", "inv-001") is True

    def test_whitespace_normalized(self):
        assert exact_match("  hello world  ", "hello world") is True

    def test_punctuation_stripped(self):
        assert exact_match("hello, world!", "hello world") is True

    def test_mismatch(self):
        assert exact_match("INV-001", "INV-002") is False

    def test_both_none(self):
        assert exact_match(None, None) is True

    def test_one_none(self):
        assert exact_match(None, "value") is False

    def test_numeric(self):
        assert exact_match(100.0, "100.0") is True


class TestTokenF1:
    def test_perfect_match(self):
        assert token_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("foo bar", "baz qux") == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = token_f1("hello world foo", "hello world bar")
        assert 0.0 < score < 1.0

    def test_both_empty(self):
        assert token_f1("", "") == pytest.approx(1.0)

    def test_one_empty(self):
        assert token_f1("", "hello") == pytest.approx(0.0)

    def test_subset(self):
        score = token_f1("hello", "hello world")
        assert score == pytest.approx(2 / 3, rel=1e-3)


class TestPartialMatch:
    def test_exact(self):
        assert partial_match("hello world", "hello world") is True

    def test_substring(self):
        assert partial_match("Acme Corp Ltd", "Acme Corp") is True

    def test_unrelated(self):
        assert partial_match("xyz", "abc 123") is False

    def test_both_none(self):
        assert partial_match(None, None) is True

    def test_threshold(self):
        # 50% similarity — below default threshold of 80
        assert partial_match("aaaa", "bbbb") is False


class TestComputeFieldMetrics:
    def test_returns_all_keys(self):
        result = compute_field_metrics("value", "value")
        assert set(result.keys()) == {"exact_match", "token_f1", "partial_match"}

    def test_perfect_score(self):
        result = compute_field_metrics("acme corp", "acme corp")
        assert result["exact_match"] == 1.0
        assert result["token_f1"] == 1.0
        assert result["partial_match"] == 1.0


class TestAggregateMetrics:
    def test_average(self):
        data = [
            {"exact_match": 1.0, "token_f1": 0.8, "partial_match": 1.0},
            {"exact_match": 0.0, "token_f1": 0.4, "partial_match": 0.0},
        ]
        agg = aggregate_metrics(data)
        assert agg["exact_match"] == pytest.approx(0.5)
        assert agg["token_f1"] == pytest.approx(0.6)

    def test_empty(self):
        agg = aggregate_metrics([])
        assert agg["exact_match"] == 0.0
