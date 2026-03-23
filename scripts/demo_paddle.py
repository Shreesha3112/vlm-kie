#!/usr/bin/env python
"""End-to-end demo for all Paddle-based models.

Usage:
    uv run python scripts/demo_paddle.py --image path/to/invoice.png
    uv run python scripts/demo_paddle.py           # auto-downloads sample
    uv run python scripts/demo_paddle.py --skip-vl # skip 7B VLM model
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import rich.logging
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logging.basicConfig(
    level=logging.INFO,
    handlers=[rich.logging.RichHandler(show_path=False)],
)
logger = logging.getLogger(__name__)
console = Console()

# Free sample invoice URL (public domain / openly licensed)
_SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Invoice_template.jpg/640px-Invoice_template.jpg"
_SAMPLE_PATH = Path("outputs/paddle_demo/sample_invoice.jpg")

# Minimal schema passed to extract() — only used by LLM-based backends
_SCHEMA: dict = {
    "fields": {
        "invoice_number": {"type": "string", "description": "Invoice number"},
        "invoice_date": {"type": "string", "description": "Invoice date"},
        "vendor_name": {"type": "string", "description": "Vendor / seller name"},
        "total": {"type": "number", "description": "Total amount due"},
    },
    "prompt_templates": {},
}


def download_sample(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        console.print(f"[dim]Using cached sample: {dest}[/dim]")
        return dest
    console.print(f"[cyan]Downloading sample invoice → {dest}[/cyan]")
    urllib.request.urlretrieve(_SAMPLE_URL, dest)  # noqa: S310
    console.print("[green]Download complete.[/green]")
    return dest


def run_model(model_id: str, image: Image.Image) -> tuple[str, float]:
    """Load model, run extract(), unload. Returns (output, elapsed_seconds)."""
    from vlm_kie.models.registry import build_model  # noqa: PLC0415

    model = build_model(model_id)
    t0 = time.perf_counter()
    model.load()
    output = model.extract(image, _SCHEMA)
    elapsed = time.perf_counter() - t0
    model.unload()
    return output, elapsed


def print_result(model_id: str, output: str, elapsed: float) -> None:
    char_count = len(output)
    header = f"[bold cyan]{model_id}[/bold cyan]  [{elapsed:.1f}s | {char_count} chars]"
    preview = output[:800] + ("…" if len(output) > 800 else "")
    console.print(Panel(preview, title=header, border_style="blue"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paddle model end-to-end demo.")
    parser.add_argument("--image", default=None, help="Path to invoice/receipt image")
    parser.add_argument(
        "--skip-vl",
        action="store_true",
        help="Skip PaddleOCR-VL-1.5 (7B download)",
    )
    args = parser.parse_args()

    # --- Resolve image ---
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            console.print(f"[red]Image not found: {img_path}[/red]")
            sys.exit(1)
    else:
        img_path = download_sample(_SAMPLE_PATH)

    image = Image.open(img_path).convert("RGB")
    console.print(f"\n[bold]Image:[/bold] {img_path}  ({image.size[0]}×{image.size[1]})\n")

    # --- Model list ---
    models_to_run = [
        "pp-ocr-v5",
        "pp-structure-v3",
        "pp-chatocrv4",
    ]
    if not args.skip_vl:
        models_to_run.append("paddleocr-vl-1.5")

    # --- Summary table ---
    table = Table(title="Paddle Demo Results", show_lines=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Time (s)", justify="right")
    table.add_column("Chars", justify="right")
    table.add_column("Status", justify="center")

    for model_id in models_to_run:
        console.rule(f"[bold]{model_id}[/bold]")
        try:
            output, elapsed = run_model(model_id, image)
            print_result(model_id, output, elapsed)
            table.add_row(model_id, f"{elapsed:.1f}", str(len(output)), "[green]OK[/green]")
        except Exception as exc:
            console.print(f"[red]ERROR ({model_id}): {exc}[/red]")
            logger.exception("Model %s failed", model_id)
            table.add_row(model_id, "-", "-", "[red]FAIL[/red]")

    console.print("\n")
    console.print(table)
    console.print("\n[bold green]Demo complete.[/bold green]")
    console.print(f"Structured outputs saved to: [cyan]outputs/paddle_demo/[/cyan]")


if __name__ == "__main__":
    main()
