#!/usr/bin/env python
"""Entry point for single-image extraction or full benchmark.

Usage:
    uv run run.py --model qwen3.5-2b --image path/to/invoice.png
    uv run run.py --model all --dataset cord-v2 --n 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import rich.logging
from rich.console import Console

logging.basicConfig(
    level=logging.INFO,
    handlers=[rich.logging.RichHandler(show_path=False)],
)
logger = logging.getLogger(__name__)
console = Console()


def run_single(model_id: str, image_path: str) -> None:
    from vlm_kie.models.registry import build_model
    from vlm_kie.pipelines.extractor import load_extraction_schema, run_extraction

    schema = load_extraction_schema()
    model = build_model(model_id)
    model.load()
    try:
        result = run_extraction(model, image_path, schema)
    finally:
        model.unload()

    if result.error:
        console.print(f"[red]Error:[/red] {result.error}")
        console.print(f"[yellow]Raw output:[/yellow] {result.raw_output[:500]}")
    else:
        console.print_json(result.model_dump_json(exclude={"raw_output"}))


def run_full_benchmark(model_id: str, dataset: str, n: int) -> None:
    """Delegate to scripts/run_benchmark.py logic."""
    from vlm_kie.data.loader import load_cord_v2
    from vlm_kie.eval.report import write_comparison_md, write_metrics_json
    from vlm_kie.models.registry import build_model, list_model_ids
    from vlm_kie.pipelines.batch import create_run_dir, run_batch
    from vlm_kie.pipelines.extractor import load_extraction_schema

    import pandas as pd

    model_ids = list_model_ids() if model_id == "all" else [model_id]
    samples = load_cord_v2(n=n)
    image_paths = [s["image_path"] for s in samples]
    ground_truths = [s["ground_truth"] for s in samples]
    schema = load_extraction_schema()
    run_dir = create_run_dir()
    console.print(f"[bold green]Run directory:[/bold green] {run_dir}")

    for mid in model_ids:
        console.rule(f"[bold blue]{mid}")
        model = build_model(mid)
        model.load()
        try:
            results = run_batch(model, image_paths, run_dir, schema)
            write_metrics_json(mid, results, ground_truths, run_dir)
        finally:
            model.unload()

    rows = []
    for mid in model_ids:
        mp = run_dir / mid / "metrics.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            rows.append({"model": mid, **m})

    if rows:
        df = pd.DataFrame(rows)
        md_path = write_comparison_md(run_dir, df)
        console.print(f"\n[bold green]Report:[/bold green] {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Key Information Extraction")
    parser.add_argument("--model", required=True, help="Model ID or 'all'")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="Path to a single invoice image")
    mode.add_argument("--dataset", choices=["cord-v2"], help="Run full benchmark")

    parser.add_argument("--n", type=int, default=100, help="Samples for dataset mode")
    args = parser.parse_args()

    if args.image:
        run_single(args.model, args.image)
    else:
        run_full_benchmark(args.model, args.dataset, args.n)


if __name__ == "__main__":
    main()
