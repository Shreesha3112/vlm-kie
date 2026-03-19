"""GLM-OCR via HuggingFace transformers — document-specific backend."""

from __future__ import annotations

import logging
from typing import Any

from PIL.Image import Image as PILImage

from vlm_kie.models.base import BaseVLM

logger = logging.getLogger(__name__)

HF_ID = "zai-org/GLM-OCR"


class GLMOCRBackend(BaseVLM):
    """GLM-OCR 0.9B — document-optimized model loaded at fp16.

    Fits in ~1.5GB VRAM without quantization.
    Uses AutoModelForImageTextToText + AutoProcessor from transformers.
    """

    def __init__(self, model_id: str, hf_id: str = HF_ID, **kwargs: Any) -> None:
        self.model_id = model_id
        self.hf_id = hf_id
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        from vlm_kie.utils.device import get_device  # noqa: PLC0415

        device = get_device()
        logger.info("Loading %s on %s", self.hf_id, device)

        self._processor = AutoProcessor.from_pretrained(
            self.hf_id, trust_remote_code=True
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.hf_id,
            torch_dtype="float16",
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("GLM-OCR loaded.")

    def _build_bbox_prompt(self, schema: dict[str, Any]) -> str:
        """Build a GLM-OCR-specific prompt requesting value+bbox per field."""
        fields = schema.get("fields", {})
        scalar_fields = {k: v for k, v in fields.items() if k != "line_items"}
        scalar_example = ", ".join(
            f'"{k}": {{"value": <{v.get("type", "string")}>, "bbox": [x1, y1, x2, y2]}}'
            for k, v in scalar_fields.items()
        )
        return (
            "Extract all invoice fields from this document image. "
            "For every field, return both its value and the bounding box [x1, y1, x2, y2] "
            "(pixel coordinates, top-left to bottom-right) of the text region in the image.\n\n"
            "Return a single valid JSON object in this exact format:\n"
            "{\n"
            f"  {scalar_example},\n"
            '  "line_items": [\n'
            "    {\n"
            '      "description": {"value": <string>, "bbox": [x1, y1, x2, y2]},\n'
            '      "quantity": {"value": <number>, "bbox": [x1, y1, x2, y2]},\n'
            '      "unit_price": {"value": <number>, "bbox": [x1, y1, x2, y2]},\n'
            '      "total": {"value": <number>, "bbox": [x1, y1, x2, y2]}\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Use null for missing fields. Return only valid JSON, no extra text."
        )

    def extract(self, image: PILImage, schema: dict[str, Any]) -> str:
        if self._model is None:
            self.load()

        prompt = self._build_bbox_prompt(schema)

        # GLM-OCR uses a conversational format with image + text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self._model.device)

        import torch  # noqa: PLC0415

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        return self._processor.decode(generated, skip_special_tokens=True)

    def unload(self) -> None:
        import torch  # noqa: PLC0415

        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GLM-OCR unloaded.")
