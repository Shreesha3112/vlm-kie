"""PaddleOCR-VL-1.5 via HuggingFace transformers — document-specific backend."""

from __future__ import annotations

import logging
from typing import Any

from PIL.Image import Image as PILImage

from vlm_kie.models.base import BaseVLM

logger = logging.getLogger(__name__)

HF_ID = "PaddlePaddle/PaddleOCR-VL-1.5"


class PaddleOCRVLBackend(BaseVLM):
    """PaddleOCR-VL-1.5 (~7B params) loaded via transformers.

    Uses 4-bit BitsAndBytes quantization to fit in 4GB VRAM.
    Falls back to 8-bit if 4-bit is unavailable.
    Does NOT require the paddlepaddle package.
    """

    def __init__(self, model_id: str, hf_id: str = HF_ID, **kwargs: Any) -> None:
        self.model_id = model_id
        self.hf_id = hf_id
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        from transformers import (  # noqa: PLC0415
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        import torch  # noqa: PLC0415

        from vlm_kie.utils.device import get_device  # noqa: PLC0415

        device = get_device()
        logger.info("Loading %s on %s (4-bit quantized)", self.hf_id, device)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self._processor = AutoProcessor.from_pretrained(
            self.hf_id, trust_remote_code=True
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.hf_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("PaddleOCR-VL-1.5 loaded.")

    def extract(self, image: PILImage, schema: dict[str, Any]) -> str:
        if self._model is None:
            self.load()

        prompt = self._build_json_prompt(schema)

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
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self._model.device)

        import torch  # noqa: PLC0415

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

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
        logger.info("PaddleOCR-VL-1.5 unloaded.")
