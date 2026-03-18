"""PP-ChatOCRv4 hybrid pipeline — PaddleX OCR + Qwen3.5 for structured extraction."""

from __future__ import annotations

import logging
from typing import Any

from PIL.Image import Image as PILImage

from vlm_kie.models.base import BaseVLM

logger = logging.getLogger(__name__)


class PPChatOCRv4Backend(BaseVLM):
    """Hybrid pipeline:
    1. PaddleX PP-ChatOCRv4-doc extracts raw OCR text from the image.
    2. QwenOllamaBackend (qwen3.5:2b) structures the OCR text into JSON.

    Requires: uv sync --extra paddle
    """

    def __init__(self, model_id: str, qwen_tag: str = "qwen3.5:2b", **kwargs: Any) -> None:
        self.model_id = model_id
        self.qwen_tag = qwen_tag
        self._pipeline: Any = None
        self._qwen: Any = None

    def load(self) -> None:
        try:
            import paddlex  # noqa: PLC0415

            logger.info("Creating PP-ChatOCRv4-doc pipeline via PaddleX...")
            self._pipeline = paddlex.create_pipeline("PP-ChatOCRv4-doc")
            logger.info("PP-ChatOCRv4-doc pipeline ready.")
        except ImportError as exc:
            raise ImportError(
                "PaddleX not installed. Install with: uv sync --extra paddle"
            ) from exc

        from vlm_kie.models.qwen_ollama import QwenOllamaBackend  # noqa: PLC0415

        self._qwen = QwenOllamaBackend(
            model_id="qwen3.5-2b-internal",
            ollama_tag=self.qwen_tag,
        )
        self._qwen.load()

    def extract(self, image: PILImage, schema: dict[str, Any]) -> str:
        if self._pipeline is None or self._qwen is None:
            self.load()

        # Step 1: OCR text extraction via PaddleX
        import tempfile  # noqa: PLC0415
        import os  # noqa: PLC0415

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        try:
            result = self._pipeline.visual_predict(tmp_path)
            ocr_text = self._flatten_ocr_result(result)
        finally:
            os.unlink(tmp_path)

        # Step 2: Structure OCR text into JSON via Qwen
        fields = schema.get("fields", {})
        field_names = ", ".join(fields.keys())
        template = schema.get("prompt_templates", {}).get("default", "")

        if template:
            field_lines = []
            for name, meta in fields.items():
                desc = meta.get("description", "")
                field_type = meta.get("type", "string")
                field_lines.append(f'  "{name}": ({field_type}) {desc}')
            field_list = "\n".join(field_lines)
            prompt = template.format(field_list=field_list, field_names=field_names)
        else:
            prompt = f"Extract these invoice fields as JSON: {field_names}"

        text_prompt = (
            f"OCR extracted text from invoice:\n\n{ocr_text}\n\n"
            f"{prompt}"
        )

        # Use Qwen text-only for structuring (no image needed here)
        import ollama  # noqa: PLC0415

        response = ollama.chat(
            model=self.qwen_tag,
            messages=[{"role": "user", "content": text_prompt}],
        )
        return response["message"]["content"]

    def _flatten_ocr_result(self, result: Any) -> str:
        """Extract plain text from PaddleX pipeline result."""
        try:
            # PaddleX returns a generator/list of prediction dicts
            texts = []
            for item in result:
                if hasattr(item, "rec_texts"):
                    texts.extend(item.rec_texts)
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, list):
                            texts.extend(str(x) for x in v)
            return "\n".join(texts) if texts else str(result)
        except Exception:
            return str(result)

    def unload(self) -> None:
        self._pipeline = None
        if self._qwen:
            self._qwen.unload()
            self._qwen = None
        logger.info("PP-ChatOCRv4 pipeline unloaded.")
