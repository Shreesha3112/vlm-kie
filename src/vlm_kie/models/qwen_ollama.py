"""Qwen3.5 via Ollama — general VLM backend."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from PIL.Image import Image as PILImage

from vlm_kie.models.base import BaseVLM

logger = logging.getLogger(__name__)


class QwenOllamaBackend(BaseVLM):
    """Qwen3.5 inference via the Ollama Python client.

    Ollama manages model lifecycle and quantization automatically.
    No GPU memory management needed here.
    """

    def __init__(self, model_id: str, ollama_tag: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.ollama_tag = ollama_tag  # e.g. "qwen3.5:2b"
        self._client: Any = None

    def load(self) -> None:
        """Verify Ollama is reachable and model is available."""
        try:
            import ollama  # noqa: PLC0415

            self._client = ollama
            # Ping Ollama — will raise if server not running
            models = ollama.list()
            available = [m.model for m in models.models]
            if self.ollama_tag not in available:
                logger.warning(
                    "Model %s not found locally. Run: ollama pull %s",
                    self.ollama_tag,
                    self.ollama_tag,
                )
            else:
                logger.info("Ollama model ready: %s", self.ollama_tag)
        except Exception as exc:
            raise RuntimeError(
                f"Ollama not available. Start with: ollama serve\n{exc}"
            ) from exc

    def extract(self, image: PILImage, schema: dict[str, Any]) -> str:
        """Encode image as base64 and send to Ollama chat API."""
        if self._client is None:
            self.load()

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = self._build_json_prompt(schema)

        response = self._client.chat(
            model=self.ollama_tag,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
        )
        return response["message"]["content"]

    def unload(self) -> None:
        """Ollama manages model memory — nothing to do here."""
        self._client = None
