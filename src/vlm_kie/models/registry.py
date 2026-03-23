"""Model registry: load model configs from models.yaml and instantiate backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from vlm_kie.models.base import BaseVLM

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_model_configs() -> list[dict[str, Any]]:
    """Return all model configs from models.yaml."""
    with open(_CONFIG_DIR / "models.yaml") as f:
        data = yaml.safe_load(f)
    return data["models"]


def get_model_config(model_id: str) -> dict[str, Any]:
    """Return config for a specific model ID."""
    for cfg in load_model_configs():
        if cfg["id"] == model_id:
            return cfg
    raise ValueError(
        f"Unknown model: {model_id!r}. "
        f"Available: {[c['id'] for c in load_model_configs()]}"
    )


def build_model(model_id: str) -> BaseVLM:
    """Instantiate the appropriate backend for the given model_id."""
    cfg = get_model_config(model_id)
    backend = cfg["backend"]

    if backend == "qwen_ollama":
        from vlm_kie.models.qwen_ollama import QwenOllamaBackend  # noqa: PLC0415

        return QwenOllamaBackend(model_id=cfg["id"], ollama_tag=cfg["ollama_tag"])

    elif backend == "glm_ocr":
        from vlm_kie.models.glm_ocr import GLMOCRBackend  # noqa: PLC0415

        return GLMOCRBackend(model_id=cfg["id"], hf_id=cfg.get("hf_id", "zai-org/GLM-OCR"))

    elif backend == "paddleocr_vl":
        from vlm_kie.models.paddleocr_vl import PaddleOCRVLBackend  # noqa: PLC0415

        return PaddleOCRVLBackend(
            model_id=cfg["id"],
            hf_id=cfg.get("hf_id", "PaddlePaddle/PaddleOCR-VL-1.5"),
        )

    elif backend == "pp_chatocrv4":
        from vlm_kie.models.pp_chatocrv4 import PPChatOCRv4Backend  # noqa: PLC0415

        return PPChatOCRv4Backend(model_id=cfg["id"])

    elif backend == "pp_ocr_v5":
        from vlm_kie.models.pp_ocr_v5 import PPOCRv5Backend  # noqa: PLC0415

        return PPOCRv5Backend(model_id=cfg["id"])

    elif backend == "pp_structure_v3":
        from vlm_kie.models.pp_structure_v3 import PPStructureV3Backend  # noqa: PLC0415

        return PPStructureV3Backend(model_id=cfg["id"])

    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def list_model_ids() -> list[str]:
    """Return all registered model IDs."""
    return [cfg["id"] for cfg in load_model_configs()]
