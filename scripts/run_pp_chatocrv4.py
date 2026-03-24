"""Run the real PP-ChatOCRv4 pipeline on a sample image.

Usage:
    uv run scripts/run_pp_chatocrv4.py [--image PATH] [--keys KEY1,KEY2,...] [--mllm] [--rag]

Defaults:
    --image   outputs/ppocr_v5_annotated/test_sample/scanned_image_annotated.png
              (falls back to src/vlm_kie/data/samples/test_sample/scanned_image.png)
    --keys    auto-selected generic document keys
    --llm     qwen2.5:7b (Ollama, GPU)
    --mllm    disabled by default (enable with --mllm, uses qwen2.5vl:7b via Ollama)
    --rag     disabled by default (enable with --rag, uses nomic-embed-text via Ollama)

Prerequisites:
    1. uv sync --extra paddle
    2. ollama serve  (with the LLM model pulled: ollama pull qwen2.5:7b)
    3. If --mllm: ollama pull qwen2.5vl:7b
    4. If --rag:  ollama pull nomic-embed-text
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "src" / "vlm_kie" / "data" / "samples" / "test_sample" / "scanned_image.png"
OUTPUT_DIR = REPO_ROOT / "outputs" / "pp_chatocrv4"

# Generic keys suitable for any scanned document
DEFAULT_KEYS = [
    "document_type",
    "date",
    "total_amount",
    "vendor_name",
    "invoice_number",
    "items",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PP-ChatOCRv4 on a sample image")
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to input image (default: test_sample/scanned_image.png)",
    )
    p.add_argument(
        "--keys",
        type=str,
        default=None,
        help="Comma-separated list of field keys to extract (default: generic doc keys)",
    )
    p.add_argument(
        "--llm",
        type=str,
        default="qwen2.5:7b",
        help="Ollama model name for LLM extraction (default: qwen2.5:7b)",
    )
    p.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost:11434/v1",
        help="Ollama base URL for LLM (default: http://localhost:11434/v1)",
    )
    p.add_argument(
        "--mllm",
        action="store_true",
        help="Enable MLLM stage (requires multimodal Ollama model, e.g. qwen2.5vl:7b)",
    )
    p.add_argument(
        "--mllm-model",
        type=str,
        default="qwen2.5vl:7b",
        help="Ollama MLLM model name (default: qwen2.5vl:7b)",
    )
    p.add_argument(
        "--rag",
        action="store_true",
        help="Enable RAG/vector stage (requires embedding Ollama model, e.g. nomic-embed-text)",
    )
    p.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model name (default: nomic-embed-text)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PaddleX visual pipeline device: cpu or gpu (default: cpu)",
    )
    return p.parse_args()


def resolve_image(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            logger.error("Image not found: %s", path)
            sys.exit(1)
        return path
    # Default: test_sample source image
    candidates = [
        DEFAULT_IMAGE,
        REPO_ROOT / "outputs" / "ppocr_v5_annotated" / "test_sample" / "scanned_image_annotated.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    logger.error("Could not find default test image. Provide --image PATH.")
    sys.exit(1)


def main() -> None:
    args = parse_args()

    image_path = resolve_image(args.image)
    key_list = [k.strip() for k in args.keys.split(",")] if args.keys else DEFAULT_KEYS

    logger.info("Image:   %s", image_path)
    logger.info("Keys:    %s", key_list)
    logger.info("LLM:     %s @ %s  (GPU via Ollama)", args.llm, args.llm_url)
    logger.info("MLLM:    %s", f"{args.mllm_model} (enabled)" if args.mllm else "disabled")
    logger.info("RAG:     %s", f"{args.embed_model} (enabled)" if args.rag else "disabled")
    logger.info("Device:  %s (PaddleX visual pipeline)", args.device)

    # Build pipeline
    try:
        from vlm_kie.models.pp_chatocrv4 import PPChatOCRv4Backend  # noqa: PLC0415
    except ImportError as exc:
        logger.error("Import failed: %s", exc)
        sys.exit(1)

    pipeline = PPChatOCRv4Backend(
        model_id="pp-chatocrv4",
        llm_base_url=args.llm_url,
        llm_model_name=args.llm,
        use_mllm=args.mllm,
        mllm_base_url=args.llm_url,
        mllm_model_name=args.mllm_model,
        use_rag=args.rag,
        embed_base_url=args.llm_url,
        embed_model_name=args.embed_model,
        paddle_device=args.device,
    )

    logger.info("\n=== Running PP-ChatOCRv4 pipeline ===\n")
    try:
        result = pipeline.predict_file(str(image_path), key_list)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    out_json = OUTPUT_DIR / f"{stem}_chatocrv4.json"

    payload = {
        "source": str(image_path),
        "pipeline": "PP-ChatOCRv4",
        "llm": args.llm,
        "use_mllm": args.mllm,
        "use_rag": args.rag,
        "key_list": key_list,
        "result": result,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    logger.info("\n=== Results ===")
    chat_res = result.get("chat_res", result)
    if isinstance(chat_res, dict):
        for k, v in chat_res.items():
            print(f"  {k}: {v}")
    else:
        print(chat_res)

    logger.info("\nOutput saved to: %s", out_json)


if __name__ == "__main__":
    main()
