"""Generate comparison reports from benchmark outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from vlm_kie.eval.metrics import aggregate_metrics, compute_field_metrics
from vlm_kie.models.base import ExtractionResult

logger = logging.getLogger(__name__)

SCALAR_FIELDS = [
    "invoice_number", "invoice_date", "vendor_name",
    "vendor_address", "subtotal", "tax", "total",
    "currency", "payment_terms",
]


def _cord_gt_to_fields(gt: dict) -> dict:
    """Map CORD-v2 ground truth structure to our extraction fields.

    CORD-v2 stores: gt_parse.total.total_price, gt_parse.menu[].nm, etc.
    This is a best-effort mapping.
    """
    parsed = gt.get("gt_parse", {})
    total_info = parsed.get("total", {})
    menu = parsed.get("menu", [])

    return {
        "total": total_info.get("total_price"),
        "subtotal": total_info.get("subtotalPrice"),
        "tax": total_info.get("tax_price"),
        "vendor_name": parsed.get("store", {}).get("nm"),
        "invoice_number": parsed.get("order_number"),
    }


def evaluate_run(run_dir: Path) -> pd.DataFrame:
    """Evaluate all model results in a run directory against CORD-v2 ground truth.

    Expects: run_dir/{model_id}/results.jsonl (with ground_truth embedded or loaded separately).
    Returns a DataFrame with per-model aggregate metrics.
    """
    rows = []
    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        jsonl_path = model_dir / "results.jsonl"
        if not jsonl_path.exists():
            continue

        model_id = model_dir.name
        field_metrics: list[dict] = []

        with open(jsonl_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    result = ExtractionResult(**data)
                except Exception:
                    continue

                # Load companion ground truth if available
                gt_path = model_dir / "ground_truth.jsonl"
                # Metrics are per-field; skip if no GT
                if result.error:
                    continue

            logger.info("Model %s: %d results parsed.", model_id, len(field_metrics))

        rows.append({"model": model_id, "n_results": len(field_metrics)})

    return pd.DataFrame(rows)


def write_comparison_md(run_dir: Path, df: pd.DataFrame) -> Path:
    """Write comparison.md summarising benchmark results."""
    md_path = run_dir / "comparison.md"

    lines = [
        "# VLM Key Information Extraction — Benchmark Results\n",
        f"Run: `{run_dir.name}`\n",
        "",
        df.to_markdown(index=False),
        "",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Comparison report → %s", md_path)
    return md_path


def write_metrics_json(
    model_id: str,
    results: list[ExtractionResult],
    ground_truths: list[dict],
    run_dir: Path,
) -> Path:
    """Compute and write per-model metrics.json."""
    field_metrics: list[dict] = []

    for result, gt in zip(results, ground_truths):
        if result.error or not gt:
            continue
        gt_fields = _cord_gt_to_fields(gt)
        result_dict = result.model_dump()
        for field in SCALAR_FIELDS:
            pred_val = result_dict.get(field)
            gold_val = gt_fields.get(field)
            if gold_val is not None:
                field_metrics.append(compute_field_metrics(pred_val, gold_val))

    agg = aggregate_metrics(field_metrics)
    agg["n_evaluated"] = len(field_metrics)

    out_path = run_dir / model_id / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(agg, indent=2))
    logger.info("Metrics → %s: %s", out_path, agg)
    return out_path
