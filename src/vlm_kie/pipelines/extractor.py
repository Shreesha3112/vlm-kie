"""Extraction pipeline: model → raw JSON → ExtractionResult."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from PIL.Image import Image as PILImage

from vlm_kie.models.base import BaseVLM, ExtractionResult, LineItem
from vlm_kie.utils.image import load_image, resize_for_model
from vlm_kie.utils.json_repair import parse_llm_json

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_extraction_schema() -> dict[str, Any]:
    """Load extraction.yaml field schema and prompt templates."""
    schema_path = _CONFIG_DIR / "extraction.yaml"
    with open(schema_path) as f:
        return yaml.safe_load(f)


def _coerce_number(value: Any) -> float | None:
    """Try to convert value to float, return None on failure."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _parse_line_items(raw: Any) -> list[LineItem]:
    """Parse line_items from raw JSON value."""
    if not isinstance(raw, list):
        return []
    items = []
    for entry in raw:
        if isinstance(entry, dict):
            items.append(
                LineItem(
                    description=entry.get("description"),
                    quantity=_coerce_number(entry.get("quantity")),
                    unit_price=_coerce_number(entry.get("unit_price")),
                    total=_coerce_number(entry.get("total")),
                )
            )
    return items


def run_extraction(
    model: BaseVLM,
    image_path: str | Path,
    schema: dict[str, Any] | None = None,
) -> ExtractionResult:
    """Run a single extraction: load image → model.extract → parse JSON → ExtractionResult."""
    image_path = Path(image_path)

    if schema is None:
        schema = load_extraction_schema()

    result = ExtractionResult(model_id=model.model_id, image_path=str(image_path))

    try:
        image: PILImage = load_image(image_path)
        image = resize_for_model(image)

        raw_output = model.extract(image, schema)
        result.raw_output = raw_output

        parsed = parse_llm_json(raw_output)
        if parsed is None:
            result.error = "JSON parse failed"
            return result

        if isinstance(parsed, dict):
            result.invoice_number = parsed.get("invoice_number")
            result.invoice_date = parsed.get("invoice_date")
            result.vendor_name = parsed.get("vendor_name")
            result.vendor_address = parsed.get("vendor_address")
            result.line_items = _parse_line_items(parsed.get("line_items", []))
            result.subtotal = _coerce_number(parsed.get("subtotal"))
            result.tax = _coerce_number(parsed.get("tax"))
            result.total = _coerce_number(parsed.get("total"))
            result.currency = parsed.get("currency")
            result.payment_terms = parsed.get("payment_terms")
        else:
            result.error = f"Expected JSON object, got {type(parsed).__name__}"

    except Exception as exc:
        logger.exception("Extraction failed for %s", image_path)
        result.error = str(exc)

    return result
