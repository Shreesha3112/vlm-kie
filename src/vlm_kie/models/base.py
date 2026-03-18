"""Base interface for all VLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL.Image import Image as PILImage
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    total: float | None = None


class ExtractionResult(BaseModel):
    """Structured output from a VLM extraction run."""

    model_id: str
    image_path: str

    # Core invoice fields
    invoice_number: str | None = None
    invoice_date: str | None = None
    vendor_name: str | None = None
    vendor_address: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = None
    payment_terms: str | None = None

    # Raw output for debugging / re-parsing
    raw_output: str = ""
    error: str | None = None


class BaseVLM(ABC):
    """Abstract base class for all VLM backends."""

    model_id: str

    @abstractmethod
    def load(self) -> None:
        """Load model into memory / verify availability."""
        ...

    @abstractmethod
    def extract(self, image: PILImage, schema: dict[str, Any]) -> str:
        """Run inference and return raw model output string."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
        ...

    def _build_json_prompt(self, schema: dict[str, Any]) -> str:
        """Build a structured JSON extraction prompt from the schema."""
        fields = schema.get("fields", {})
        field_lines = []
        for name, meta in fields.items():
            desc = meta.get("description", "")
            field_type = meta.get("type", "string")
            field_lines.append(f'  "{name}": ({field_type}) {desc}')
        field_list = "\n".join(field_lines)

        template = schema.get("prompt_templates", {}).get("default", "")
        if template:
            return template.format(
                field_list=field_list,
                field_names=", ".join(fields.keys()),
            )

        # Fallback minimal prompt
        return (
            f"Extract these invoice fields as JSON:\n{field_list}\n\n"
            "Return only valid JSON, null for missing fields."
        )
