"""Evaluation metrics: exact match, token F1, partial match."""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz


def _normalize(text: Any) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(pred: Any, gold: Any) -> bool:
    """Exact match after normalization."""
    return _normalize(pred) == _normalize(gold)


def token_f1(pred: Any, gold: Any) -> float:
    """Token-level F1 score (SQuAD-style).

    Returns a float in [0, 1].
    """
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    # Count token overlap
    pred_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    overlap = 0
    for t in gold_tokens:
        if pred_counts.get(t, 0) > 0:
            overlap += 1
            pred_counts[t] -= 1

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def partial_match(pred: Any, gold: Any, threshold: int = 80) -> bool:
    """Partial string match using rapidfuzz partial_ratio.

    Returns True if ratio >= threshold (0-100).
    """
    p = _normalize(pred)
    g = _normalize(gold)
    if not p and not g:
        return True
    if not p or not g:
        return False
    return fuzz.partial_ratio(p, g) >= threshold


def compute_field_metrics(pred: Any, gold: Any) -> dict[str, float]:
    """Compute all three metrics for a single field."""
    return {
        "exact_match": float(exact_match(pred, gold)),
        "token_f1": token_f1(pred, gold),
        "partial_match": float(partial_match(pred, gold)),
    }


def aggregate_metrics(per_field: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across multiple field results."""
    if not per_field:
        return {"exact_match": 0.0, "token_f1": 0.0, "partial_match": 0.0}
    keys = per_field[0].keys()
    return {k: sum(r[k] for r in per_field) / len(per_field) for k in keys}
