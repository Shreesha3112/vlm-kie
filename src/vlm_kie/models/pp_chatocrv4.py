"""PP-ChatOCRv4 hybrid pipeline — PaddleX OCR + Qwen2.5 (transformers) for structured extraction."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from PIL.Image import Image as PILImage

from vlm_kie.models.base import BaseVLM

logger = logging.getLogger(__name__)

# Small text-only model for structuring OCR output; fits in ~2GB VRAM at fp16
DEFAULT_QWEN_HF_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class PPChatOCRv4Backend(BaseVLM):
    """Hybrid pipeline:
    1. PaddleX PP-ChatOCRv4-doc extracts raw OCR text from the image (CPU).
    2. Qwen2.5-1.5B-Instruct (transformers, GPU) structures OCR text into JSON.

    Requires: uv sync --extra paddle
    """

    def __init__(self, model_id: str, qwen_hf_id: str = DEFAULT_QWEN_HF_ID, **kwargs: Any) -> None:
        self.model_id = model_id
        self.qwen_hf_id = qwen_hf_id
        self._pipeline: Any = None
        self._qwen_model: Any = None
        self._qwen_tokenizer: Any = None
        self._qwen_device: str = "cpu"

    def load(self) -> None:
        # --- PaddleOCR engine (CPU to avoid VRAM conflict) ---
        try:
            from paddleocr import PaddleOCR  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR not installed. Install with: uv sync --extra paddle"
            ) from exc

        logger.info("Initialising PaddleOCR on CPU...")
        # device="cpu", enable_mkldnn=False avoids OneDNN PIR attribute bug in PaddlePaddle 3.x
        self._pipeline = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            device="cpu",
            enable_mkldnn=False,
        )
        logger.info("PaddleOCR ready.")

        # --- Qwen text model via transformers (GPU if available) ---
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._qwen_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading %s on %s for OCR structuring...", self.qwen_hf_id, self._qwen_device)

        self._qwen_tokenizer = AutoTokenizer.from_pretrained(self.qwen_hf_id)
        self._qwen_model = AutoModelForCausalLM.from_pretrained(
            self.qwen_hf_id,
            torch_dtype=torch.float16 if self._qwen_device == "cuda" else torch.float32,
            device_map=self._qwen_device,
        )
        self._qwen_model.eval()
        logger.info("Qwen text model loaded.")

    def extract(self, image: PILImage, schema: dict[str, Any]) -> str:
        if self._pipeline is None or self._qwen_model is None:
            self.load()

        # Step 1: OCR text extraction via PaddleX
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        try:
            result = self._pipeline.predict(tmp_path)
            ocr_text = self._flatten_ocr_result(result)
        finally:
            os.unlink(tmp_path)

        logger.info("OCR extracted %d chars", len(ocr_text))

        # Step 2: Structure OCR text into JSON via Qwen
        fields = schema.get("fields", {})
        field_names = ", ".join(fields.keys())
        template = schema.get("prompt_templates", {}).get("default", "")

        if template:
            field_lines = [
                f'  "{name}": ({meta.get("type","string")}) {meta.get("description","")}'
                for name, meta in fields.items()
            ]
            prompt = template.format(
                field_list="\n".join(field_lines),
                field_names=field_names,
            )
        else:
            prompt = f"Extract these invoice fields as JSON: {field_names}"

        full_prompt = f"OCR extracted text from invoice:\n\n{ocr_text}\n\n{prompt}"

        messages = [{"role": "user", "content": full_prompt}]
        text = self._qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._qwen_tokenizer(text, return_tensors="pt").to(self._qwen_device)

        import torch  # noqa: PLC0415

        with torch.no_grad():
            output_ids = self._qwen_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        return self._qwen_tokenizer.decode(generated, skip_special_tokens=True)

    def _flatten_ocr_result(self, result: Any) -> str:
        """Extract plain text from PaddleOCR 3.x result.

        PaddleOCR.predict() returns a list of OCRResult objects (dict-like).
        Each item has item["rec_texts"] — a list of recognized text strings.
        """
        try:
            texts: list[str] = []
            for item in result:
                if item is None:
                    continue
                # Dict-style access (OCRResult is a BaseCVResult subclass)
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
        import torch  # noqa: PLC0415

        self._pipeline = None
        del self._qwen_model
        del self._qwen_tokenizer
        self._qwen_model = None
        self._qwen_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("PP-ChatOCRv4 pipeline unloaded.")
