"""Dataset loading: HuggingFace datasets and local image folders."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).parent / "samples"
CORD_HF_ID = "naver-clova-ix/cord-v2"


def load_cord_v2(n: int = 100, split: str = "test") -> list[dict]:
    """Load n samples from CORD-v2 test split.

    Returns list of dicts with keys: image_path (Path), ground_truth (dict).
    Images are saved to src/vlm_kie/data/samples/cord-v2/.
    """
    from datasets import load_dataset  # noqa: PLC0415

    logger.info("Loading CORD-v2 (split=%s, n=%d)...", split, n)
    dataset = load_dataset(CORD_HF_ID, split=split, streaming=False)

    cord_dir = SAMPLES_DIR / "cord-v2"
    cord_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for i, item in enumerate(dataset):
        if i >= n:
            break
        img_path = cord_dir / f"{i:04d}.png"
        if not img_path.exists():
            item["image"].save(img_path, format="PNG")

        import json  # noqa: PLC0415

        try:
            gt = json.loads(item.get("ground_truth", "{}"))
        except (json.JSONDecodeError, TypeError):
            gt = {}

        samples.append({"image_path": img_path, "ground_truth": gt, "index": i})

    logger.info("Loaded %d CORD-v2 samples → %s", len(samples), cord_dir)
    return samples


def load_local_images(folder: str | Path) -> list[dict]:
    """Load all images from a local folder.

    Returns list of dicts with keys: image_path (Path), ground_truth (None).
    """
    folder = Path(folder)
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)
    logger.info("Found %d images in %s", len(image_paths), folder)
    return [{"image_path": p, "ground_truth": None} for p in image_paths]
