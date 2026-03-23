"""PP-OCR V5 backend — PaddleOCR 3.x text detection + recognition."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from PIL.Image import Image as PILImage

from vlm_kie.models.base import BaseVLM

logger = logging.getLogger(__name__)


class PPOCRv5Backend(BaseVLM):
    """PP-OCR V5 text extraction pipeline (CPU, R&D mode — returns raw OCR text).

    Requires: uv sync --extra paddle
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self._ocr: Any = None

    def load(self) -> None:
        try:
            from paddleocr import PaddleOCR  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR not installed. Install with: uv sync --extra paddle"
            ) from exc

        logger.info("Initialising PP-OCR V5 on CPU...")
        self._ocr = PaddleOCR(
            lang="en",
            device="cpu",
            ocr_version="PP-OCRv5",
            enable_mkldnn=False,
        )
        logger.info("PP-OCR V5 ready.")

    def extract(self, image: PILImage, schema: dict[str, Any]) -> str:
        if self._ocr is None:
            self.load()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        try:
            result = self._ocr.predict(tmp_path)
            return self._flatten_ocr_result(result)
        finally:
            os.unlink(tmp_path)

    def _flatten_ocr_result(self, result: Any) -> str:
        """Extract plain text from PaddleOCR 3.x result objects."""
        try:
            texts: list[str] = []
            for item in result:
                if item is None:
                    continue
                rec = item.get("rec_texts") if hasattr(item, "get") else None
                if rec is None and isinstance(item, dict):
                    rec = item.get("rec_texts")
                if isinstance(rec, list):
                    texts.extend(str(t) for t in rec if t)
            if texts:
                logger.debug("Extracted %d OCR text spans", len(texts))
                return "\n".join(texts)
            logger.warning("_flatten_ocr_result: no text found; raw result: %.300s", repr(result))
            return str(result)
        except Exception as exc:
            logger.warning("_flatten_ocr_result exception: %s", exc)
            return str(result)

    def unload(self) -> None:
        self._ocr = None
        logger.info("PP-OCR V5 pipeline unloaded.")
