"""Batch pipeline: run extraction across a dataset, save per-model JSONL results."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from vlm_kie.models.base import BaseVLM, ExtractionResult
from vlm_kie.pipelines.extractor import load_extraction_schema, run_extraction

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent.parent.parent / "outputs"


def run_batch(
    model: BaseVLM,
    image_paths: list[Path],
    run_dir: Path,
    schema: dict[str, Any] | None = None,
) -> list[ExtractionResult]:
    """Run extraction on all image_paths, write JSONL to run_dir/{model_id}/results.jsonl."""
    if schema is None:
        schema = load_extraction_schema()

    out_dir = run_dir / model.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"

    results: list[ExtractionResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"[{model.model_id}]", total=len(image_paths))

        with open(jsonl_path, "w") as f:
            for img_path in image_paths:
                result = run_extraction(model, img_path, schema)
                results.append(result)
                f.write(result.model_dump_json() + "\n")
                progress.advance(task)

    logger.info(
        "Batch done: %d results → %s", len(results), jsonl_path
    )
    return results


def create_run_dir() -> Path:
    """Create a timestamped run directory under outputs/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
