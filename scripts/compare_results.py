#!/usr/bin/env python
"""Render benchmark outputs into a markdown comparison table.

Usage:
    uv run scripts/compare_results.py --run outputs/run_20260318_143000/
    uv run scripts/compare_results.py  # uses most recent run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


def find_latest_run() -> Path | None:
    runs = sorted(OUTPUTS_DIR.glob("run_*"))
    return runs[-1] if runs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VLM benchmark results.")
    parser.add_argument("--run", default=None, help="Path to run directory")
    args = parser.parse_args()

    run_dir = Path(args.run) if args.run else find_latest_run()
    if run_dir is None or not run_dir.exists():
        console.print("[red]No run directory found.[/red]")
        sys.exit(1)

    console.print(f"[bold]Run:[/bold] {run_dir}")

    rows = []
    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text())
        rows.append({"model": model_dir.name, **m})

    if not rows:
        console.print("[yellow]No metrics.json files found in run directory.[/yellow]")
        sys.exit(0)

    df = pd.DataFrame(rows)

    # Rich table display
    table = Table(title=f"Benchmark: {run_dir.name}", show_lines=True)
    for col in df.columns:
        table.add_column(col, justify="right" if col != "model" else "left")
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            v = row[col]
            if isinstance(v, float):
                values.append(f"{v:.3f}")
            else:
                values.append(str(v))
        table.add_row(*values)

    console.print(table)

    # Write/overwrite comparison.md
    md_path = run_dir / "comparison.md"
    lines = [
        f"# VLM KIE Benchmark — {run_dir.name}\n",
        "",
        df.to_markdown(index=False),
        "",
    ]
    md_path.write_text("\n".join(lines))
    console.print(f"\n[green]Saved:[/green] {md_path}")


if __name__ == "__main__":
    main()
