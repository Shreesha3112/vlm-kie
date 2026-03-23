"""PP-Structure V3 backend — layout detection + OCR + table/formula recognition."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL.Image import Image as PILImage

from vlm_kie.models.base import BaseVLM

logger = logging.getLogger(__name__)

# Output dir for markdown/JSON saved during demo runs
_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "paddle_demo"


class PPStructureV3Backend(BaseVLM):
    """PP-Structure V3 pipeline — layout + OCR + table/formula recognition.

    R&D mode: saves markdown to outputs/paddle_demo/ and returns extracted text.

    Requires: uv sync --extra paddle
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self._pipeline: Any = None

    def load(self) -> None:
        try:
            from paddleocr import PPStructureV3  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR not installed. Install with: uv sync --extra paddle"
            ) from exc

        logger.info("Initialising PP-Structure V3 on CPU...")
        self._pipeline = PPStructureV3(
            device="cpu",
            ocr_version="PP-OCRv5",
            enable_mkldnn=False,
            use_table_recognition=True,
        )
        logger.info("PP-Structure V3 ready.")

    def extract(self, image: PILImage, schema: dict[str, Any]) -> str:
        if self._pipeline is None:
            self.load()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        try:
            result = self._pipeline.predict(tmp_path)
            return self._extract_text(result, tmp_path)
        finally:
            os.unlink(tmp_path)

    def _extract_text(self, result: Any, source_path: str) -> str:
        """Save structured output and return plain text from PP-Structure V3 result."""
        try:
            _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            texts: list[str] = []

            for item in result:
                if item is None:
                    continue

                # Save markdown and JSON for visual inspection
                try:
                    item.save_to_markdown(str(_OUTPUT_DIR))
                    logger.info("PP-Structure V3 markdown saved to %s", _OUTPUT_DIR)
                except Exception as exc:
                    logger.debug("save_to_markdown failed: %s", exc)

                # Extract text from result dict
                rec = None
                if hasattr(item, "get"):
                    rec = item.get("rec_texts")
                if rec is None and isinstance(item, dict):
                    rec = item.get("rec_texts")
                if isinstance(rec, list):
                    texts.extend(str(t) for t in rec if t)

                # Also try parsing_res_list or layout blocks if rec_texts not present
                if not rec:
                    for key in ("parsing_res_list", "layout_det_res", "overall_ocr_res"):
                        block = item.get(key) if hasattr(item, "get") else None
                        if block is not None:
                            texts.append(str(block))
                            break

            if texts:
                logger.debug("PP-Structure V3: extracted %d text spans", len(texts))
                return "\n".join(texts)

            logger.warning("_extract_text: no text found; raw result: %.300s", repr(result))
            return str(result)
        except Exception as exc:
            logger.warning("_extract_text exception: %s", exc)
            return str(result)

    def unload(self) -> None:
        self._pipeline = None
        logger.info("PP-Structure V3 pipeline unloaded.")
