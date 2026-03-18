#!/usr/bin/env python
"""CLI benchmark runner.

Usage:
    uv run scripts/run_benchmark.py --model qwen3.5-2b --dataset cord-v2 --n 100
    uv run scripts/run_benchmark.py --model all --dataset cord-v2 --n 10
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import rich.logging
from rich.console import Console

from vlm_kie.data.loader import load_cord_v2, load_local_images
from vlm_kie.eval.report import write_comparison_md, write_metrics_json
from vlm_kie.models.registry import build_model, list_model_ids
from vlm_kie.pipelines.batch import create_run_dir, run_batch
from vlm_kie.pipelines.extractor import load_extraction_schema

logging.basicConfig(
    level=logging.INFO,
    handlers=[rich.logging.RichHandler(show_path=False)],
)
logger = logging.getLogger(__name__)
console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM KIE benchmark runner.")
    parser.add_argument(
        "--model",
        default="qwen3.5-2b",
        help=f"Model ID or 'all'. Available: {list_model_ids()}",
    )
    parser.add_argument(
        "--dataset",
        default="cord-v2",
        choices=["cord-v2"],
        help="Dataset to use",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--image-dir", default=None, help="Local image folder (overrides --dataset)")
    args = parser.parse_args()

    # Determine which models to run
    if args.model == "all":
        model_ids = list_model_ids()
    else:
        model_ids = [args.model]

    # Load dataset
    if args.image_dir:
        samples = load_local_images(args.image_dir)
        ground_truths = [None] * len(samples)
    else:
        samples = load_cord_v2(n=args.n)
        ground_truths = [s["ground_truth"] for s in samples]

    image_paths = [s["image_path"] for s in samples]

    schema = load_extraction_schema()
    run_dir = create_run_dir()
    console.print(f"[bold green]Run directory:[/bold green] {run_dir}")

    for model_id in model_ids:
        console.rule(f"[bold blue]{model_id}")
        model = build_model(model_id)
        model.load()
        try:
            results = run_batch(model, image_paths, run_dir, schema)
            write_metrics_json(model_id, results, ground_truths, run_dir)
        finally:
            model.unload()

    # Write combined comparison
    import pandas as pd

    rows = []
    for model_id in model_ids:
        metrics_path = run_dir / model_id / "metrics.json"
        if metrics_path.exists():
            import json

            m = json.loads(metrics_path.read_text())
            rows.append({"model": model_id, **m})

    if rows:
        df = pd.DataFrame(rows)
        md_path = write_comparison_md(run_dir, df)
        console.print(f"\n[bold green]Comparison report:[/bold green] {md_path}")
    else:
        console.print("[yellow]No metrics to compare.[/yellow]")


if __name__ == "__main__":
    main()
