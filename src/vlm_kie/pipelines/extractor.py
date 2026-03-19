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


def _extract_value_and_bbox(raw_val: Any) -> tuple[Any, list[float] | None]:
    """Handle both plain values and {value, bbox} dicts from GLM-OCR."""
    if isinstance(raw_val, dict) and "value" in raw_val:
        return raw_val["value"], raw_val.get("bbox")
    return raw_val, None


def _parse_line_items(raw: Any) -> list[LineItem]:
    """Parse line_items from raw JSON value, handling optional bbox dicts."""
    if not isinstance(raw, list):
        return []
    items = []
    for entry in raw:
        if isinstance(entry, dict):
            desc_val, desc_bbox = _extract_value_and_bbox(entry.get("description"))
            qty_val, qty_bbox = _extract_value_and_bbox(entry.get("quantity"))
            up_val, up_bbox = _extract_value_and_bbox(entry.get("unit_price"))
            tot_val, tot_bbox = _extract_value_and_bbox(entry.get("total"))
            row_bbox = next(
                (b for b in [desc_bbox, qty_bbox, up_bbox, tot_bbox] if b), None
            )
            items.append(
                LineItem(
                    description=desc_val,
                    quantity=_coerce_number(qty_val),
                    unit_price=_coerce_number(up_val),
                    total=_coerce_number(tot_val),
                    bbox=row_bbox,
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
            scalar_fields = [
                "invoice_number", "invoice_date", "vendor_name", "vendor_address",
                "subtotal", "tax", "total", "currency", "payment_terms",
            ]
            number_fields = {"subtotal", "tax", "total"}
            for field in scalar_fields:
                val, bbox = _extract_value_and_bbox(parsed.get(field))
                coerced = _coerce_number(val) if field in number_fields else val
                setattr(result, field, coerced)
                if bbox:
                    result.field_bboxes[field] = bbox
            result.line_items = _parse_line_items(parsed.get("line_items", []))
        else:
            result.error = f"Expected JSON object, got {type(parsed).__name__}"

    except Exception as exc:
        logger.exception("Extraction failed for %s", image_path)
        result.error = str(exc)

    return result
