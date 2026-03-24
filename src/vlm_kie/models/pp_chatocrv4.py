"""Real PP-ChatOCRv4 pipeline — PaddleX PPChatOCRv4Doc with local LLM via Ollama.

Architecture (all local, GPU-accelerated via Ollama):
  1. PPChatOCRv4Doc  — visual layout analysis + OCR (PaddleX, CPU)
  2. (Optional) MLLM — multimodal understanding via OpenAI-compatible local endpoint
  3. (Optional) RAG  — vector retrieval via local embedding endpoint
  4. LLM             — structured extraction via Ollama (GPU, OpenAI-compatible API)

Requires: uv sync --extra paddle
LLM/MLLM: Ollama must be running with the configured model(s).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

from PIL.Image import Image as PILImage

from vlm_kie.models.base import BaseVLM

logger = logging.getLogger(__name__)

# Default endpoints — Ollama OpenAI-compatible API (GPU via Ollama)
DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_MODEL = "qwen2.5:7b"
DEFAULT_MLLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MLLM_MODEL = "qwen2.5vl:7b"
DEFAULT_EMBED_BASE_URL = "http://localhost:11434/v1"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


class PPChatOCRv4Backend(BaseVLM):
    """Real PP-ChatOCRv4 pipeline using PaddleX PPChatOCRv4Doc.

    Stages:
      1. visual_predict() — layout parsing + OCR + (optionally) table/seal recognition
      2. build_vector()   — embed OCR chunks into a local vector store (optional, needs RAG)
      3. mllm_pred()      — MLLM pass for visual understanding (optional)
      4. chat()           — LLM-based key-value extraction with RAG context

    LLM and MLLM are served locally via Ollama (GPU-accelerated).
    Set use_mllm=True only if a multimodal Ollama model (e.g. qwen2.5vl:7b) is available.
    Set use_rag=True only if an embedding Ollama model (e.g. nomic-embed-text) is available.
    """

    def __init__(
        self,
        model_id: str,
        llm_base_url: str = DEFAULT_LLM_BASE_URL,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        use_mllm: bool = False,
        mllm_base_url: str = DEFAULT_MLLM_BASE_URL,
        mllm_model_name: str = DEFAULT_MLLM_MODEL,
        use_rag: bool = False,
        embed_base_url: str = DEFAULT_EMBED_BASE_URL,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        paddle_device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self.llm_base_url = llm_base_url
        self.llm_model_name = llm_model_name
        self.use_mllm = use_mllm
        self.mllm_base_url = mllm_base_url
        self.mllm_model_name = mllm_model_name
        self.use_rag = use_rag
        self.embed_base_url = embed_base_url
        self.embed_model_name = embed_model_name
        self.paddle_device = paddle_device  # for PaddleX visual pipeline
        self._pipeline: Any = None

    def load(self) -> None:
        try:
            from paddleocr import PPChatOCRv4Doc  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR not installed. Run: uv sync --extra paddle"
            ) from exc

        logger.info(
            "Initialising PPChatOCRv4Doc visual pipeline on %s...", self.paddle_device
        )
        self._pipeline = PPChatOCRv4Doc(device=self.paddle_device)
        logger.info("PPChatOCRv4Doc visual pipeline ready.")
        logger.info("LLM:  %s @ %s (GPU via Ollama)", self.llm_model_name, self.llm_base_url)
        if self.use_mllm:
            logger.info(
                "MLLM: %s @ %s (GPU via Ollama)", self.mllm_model_name, self.mllm_base_url
            )
        if self.use_rag:
            logger.info(
                "RAG:  %s @ %s", self.embed_model_name, self.embed_base_url
            )

    def extract(self, image: PILImage, schema: dict[str, Any]) -> str:
        if self._pipeline is None:
            self.load()

        fields = schema.get("fields", {})
        key_list = list(fields.keys()) if fields else ["content"]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        try:
            return self._run_pipeline(tmp_path, key_list)
        finally:
            os.unlink(tmp_path)

    def predict_file(self, image_path: str, key_list: list[str]) -> dict[str, Any]:
        """Run the full PP-ChatOCRv4 pipeline on a file path directly.

        Returns the raw chat_result dict from the pipeline.
        """
        if self._pipeline is None:
            self.load()
        return self._run_pipeline_raw(image_path, key_list)

    def _run_pipeline(self, image_path: str, key_list: list[str]) -> str:
        result = self._run_pipeline_raw(image_path, key_list)
        chat_res = result.get("chat_res", result)
        try:
            return json.dumps(chat_res, ensure_ascii=False, indent=2)
        except Exception:
            return str(chat_res)

    def _run_pipeline_raw(self, image_path: str, key_list: list[str]) -> dict[str, Any]:
        # --- LLM config (Ollama, GPU) ---
        chat_bot_config = {
            "module_name": "chat_bot",
            "model_name": self.llm_model_name,
            "base_url": self.llm_base_url,
            "api_type": "openai",
            "api_key": "ollama",
        }

        # --- RAG retriever config ---
        retriever_config: dict[str, Any] | None = None
        if self.use_rag:
            retriever_config = {
                "module_name": "retriever",
                "model_name": self.embed_model_name,
                "base_url": self.embed_base_url,
                "api_type": "openai",
                "api_key": "ollama",
            }

        # --- Step 1: Visual prediction (layout + OCR) ---
        logger.info("PP-ChatOCRv4 | Step 1: visual_predict(%s)", image_path)
        visual_predict_res = self._pipeline.visual_predict(
            input=image_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_common_ocr=True,
            use_seal_recognition=False,
            use_table_recognition=True,
        )
        visual_info_list: list[Any] = []
        for res in visual_predict_res:
            visual_info_list.append(res["visual_info"])
        logger.info("PP-ChatOCRv4 | visual_predict done: %d page(s)", len(visual_info_list))

        # --- Step 2: (Optional) Build RAG vector index ---
        vector_info = None
        if self.use_rag and retriever_config is not None:
            logger.info("PP-ChatOCRv4 | Step 2: build_vector (RAG)")
            vector_info = self._pipeline.build_vector(
                visual_info_list,
                flag_save_bytes_vector=False,
                retriever_config=retriever_config,
            )
            logger.info("PP-ChatOCRv4 | build_vector done")

        # --- Step 3: (Optional) MLLM visual understanding ---
        mllm_predict_info = None
        if self.use_mllm:
            mllm_config = {
                "module_name": "chat_bot",
                "model_name": self.mllm_model_name,
                "base_url": self.mllm_base_url,
                "api_type": "openai",
                "api_key": "ollama",
            }
            logger.info(
                "PP-ChatOCRv4 | Step 3: mllm_pred for keys: %s", key_list
            )
            mllm_res = self._pipeline.mllm_pred(
                input=image_path,
                key_list=key_list,
                mllm_chat_bot_config=mllm_config,
            )
            mllm_predict_info = mllm_res.get("mllm_res")
            logger.info("PP-ChatOCRv4 | mllm_pred done")

        # --- Step 4: LLM chat extraction (GPU via Ollama) ---
        logger.info(
            "PP-ChatOCRv4 | Step 4: chat() with LLM=%s for keys: %s",
            self.llm_model_name,
            key_list,
        )
        chat_result = self._pipeline.chat(
            key_list=key_list,
            visual_info=visual_info_list,
            vector_info=vector_info,
            mllm_predict_info=mllm_predict_info,
            chat_bot_config=chat_bot_config,
            retriever_config=retriever_config,
        )
        logger.info("PP-ChatOCRv4 | chat() done: %s", chat_result)
        return chat_result if isinstance(chat_result, dict) else {"chat_res": chat_result}

    def unload(self) -> None:
        self._pipeline = None
        logger.info("PPChatOCRv4 pipeline unloaded.")
