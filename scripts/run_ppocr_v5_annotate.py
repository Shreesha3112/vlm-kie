"""Run PP-OCR V5 on all sample images, output JSON with bboxes and annotated images."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalar/array types as plain Python types."""

    def default(self, obj: Any) -> Any:
        try:
            import numpy as np  # noqa: PLC0415
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)

# ── Directories ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_ROOT = REPO_ROOT / "src" / "vlm_kie" / "data" / "samples"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "ppocr_v5_annotated"

SAMPLE_DIRS = [
     SAMPLES_ROOT / "test_sample",
    #SAMPLES_ROOT / "cord-v2",
    #SAMPLES_ROOT / "doc_samples",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


# ── OCR helpers ────────────────────────────────────────────────────────────────

def load_ocr():
    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PaddleOCR not installed. Run: uv sync --extra paddle"
        ) from exc

    logger.info("Initialising PP-OCR V5 on CPU …")
    ocr = PaddleOCR(
        lang="en",
        device="cpu",
        ocr_version="PP-OCRv5",
        enable_mkldnn=False,
    )
    logger.info("PP-OCR V5 ready.")
    return ocr


def extract_bbox_data(result: Any) -> list[dict]:
    """
    Parse PaddleOCR 3.x result objects and return a list of
    { "text": str, "score": float, "bbox": [x1,y1,x2,y2] or None,
      "polygon": [[x,y], ...] or None }
    """
    records: list[dict] = []
    if result is None:
        return records

    for item in result:
        if item is None:
            continue

        # PaddleOCR 3.x result is a dict-like object
        if not hasattr(item, "get"):
            logger.debug("Unexpected item type: %s", type(item))
            continue

        texts: list[str] = item.get("rec_texts") or []
        scores: list[float] = item.get("rec_scores") or []
        polys: Any = item.get("dt_polys")   # shape: (N, 4, 2) numpy or list
        boxes: Any = item.get("rec_boxes")  # shape: (N, 4) xyxy or (N, 4, 2) poly

        # Normalise polygons to list-of-list
        def to_python(arr: Any) -> list | None:
            if arr is None:
                return None
            try:
                return arr.tolist()
            except AttributeError:
                return list(arr) if hasattr(arr, "__iter__") else None

        polys_list = to_python(polys)
        boxes_list = to_python(boxes)

        n = len(texts)
        for i in range(n):
            text = str(texts[i]) if i < len(texts) else ""
            score = float(scores[i]) if i < len(scores) else 0.0

            polygon = None
            bbox = None

            # Prefer dt_polys (polygon points)
            if polys_list is not None and i < len(polys_list):
                poly = polys_list[i]
                polygon = poly
                # Compute axis-aligned bbox from polygon
                try:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                except Exception:
                    pass

            # Fall back to rec_boxes if no dt_polys
            if bbox is None and boxes_list is not None and i < len(boxes_list):
                box = boxes_list[i]
                if isinstance(box[0], (list, tuple)):
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    polygon = box
                else:
                    # Already [x1, y1, x2, y2]
                    bbox = [float(v) for v in box[:4]]

            records.append({
                "text": text,
                "score": score,
                "bbox": bbox,
                "polygon": polygon,
            })

    return records


# ── Drawing helpers ────────────────────────────────────────────────────────────

# Color palette for bounding boxes (cycles through)
COLORS = [
    "#FF4444", "#44BB44", "#4488FF", "#FF8800",
    "#AA44FF", "#00CCCC", "#FF44AA", "#BBBB00",
]


def _get_font(size: int = 14):
    """Return a PIL font (falls back to default if no TTF available)."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()


def annotate_image(image: Image.Image, records: list[dict]) -> Image.Image:
    """Draw bounding boxes + labels on a copy of *image*."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = _get_font(max(12, img.height // 80))

    for idx, rec in enumerate(records):
        color = COLORS[idx % len(COLORS)]
        text = rec["text"]
        score = rec["score"]

        polygon = rec.get("polygon")
        bbox = rec.get("bbox")

        if polygon is not None and len(polygon) >= 3:
            # Draw filled semi-transparent polygon
            flat = [float(coord) for pt in polygon for coord in pt]
            draw.polygon(flat, outline=color, fill=color + "33")
        elif bbox:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2, fill=color + "22")

        # Label
        if bbox:
            label = f"{text[:30]} ({score:.2f})"
            x1, y1 = float(bbox[0]), float(bbox[1])
            tx, ty = max(0, x1), max(0, y1 - 16)
            # Background for label
            try:
                bb = font.getbbox(label)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
            except AttributeError:
                tw, th = len(label) * 7, 14
            draw.rectangle([tx, ty, tx + tw + 4, ty + th + 4], fill=color + "CC")
            draw.text((tx + 2, ty + 2), label, fill="white", font=font)

    return img


# ── Main pipeline ──────────────────────────────────────────────────────────────

def collect_images() -> list[Path]:
    images: list[Path] = []
    for d in SAMPLE_DIRS:
        if not d.exists():
            logger.warning("Sample dir not found: %s", d)
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                images.append(f)
    return images


def process_image(ocr, image_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run OCR
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = Image.open(image_path).convert("RGB")
        img.save(tmp.name, format="PNG")
        tmp_path = tmp.name

    try:
        result = ocr.predict(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Parse results
    records = extract_bbox_data(result)
    logger.info("  %s → %d text regions", image_path.name, len(records))

    # Save JSON
    json_path = out_dir / (image_path.stem + "_ocr.json")
    payload = {
        "source": str(image_path),
        "model": "PP-OCR V5",
        "regions": records,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, cls=_NumpyEncoder))

    # Save annotated image
    annotated = annotate_image(img, records)
    ann_path = out_dir / (image_path.stem + "_annotated.png")
    annotated.save(ann_path)

    return {"json": json_path, "annotated": ann_path, "regions": len(records)}


def main() -> None:
    images = collect_images()
    if not images:
        logger.error("No images found under %s", SAMPLES_ROOT)
        sys.exit(1)

    logger.info("Found %d images across %d sample dirs", len(images), len(SAMPLE_DIRS))

    ocr = load_ocr()

    summary: list[dict] = []
    for img_path in images:
        folder_tag = img_path.parent.name  # cord-v2 or doc_samples
        out_dir = OUTPUT_ROOT / folder_tag
        logger.info("Processing [%s] %s …", folder_tag, img_path.name)
        try:
            info = process_image(ocr, img_path, out_dir)
            summary.append({
                "image": img_path.name,
                "folder": folder_tag,
                "regions": info["regions"],
                "json_output": str(info["json"]),
                "annotated_image": str(info["annotated"]),
            })
            logger.info("  → annotated: %s", info["annotated"])
        except Exception as exc:
            logger.error("  FAILED %s: %s", img_path.name, exc, exc_info=True)
            summary.append({"image": img_path.name, "folder": folder_tag, "error": str(exc)})

    # Master summary JSON
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("\n=== Done ===")
    logger.info("Outputs in: %s", OUTPUT_ROOT)
    logger.info("Summary:    %s", summary_path)

    # Print table
    print("\nResults:")
    print(f"{'Folder':<15} {'Image':<45} {'Regions':>7}  Annotated?")
    print("-" * 80)
    for s in summary:
        if "error" in s:
            print(f"{s['folder']:<15} {s['image']:<45} {'ERROR':>7}  {s['error']}")
        else:
            annotated_ok = Path(s["annotated_image"]).exists()
            print(f"{s['folder']:<15} {s['image']:<45} {s['regions']:>7}  {'YES' if annotated_ok else 'NO'}")


if __name__ == "__main__":
    main()
