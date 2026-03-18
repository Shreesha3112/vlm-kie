"""Image loading and preprocessing utilities."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image
from PIL.Image import Image as PILImage


def load_image(path: str | Path) -> PILImage:
    """Load image from path, convert to RGB."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_for_model(image: PILImage, max_pixels: int = 1_000_000) -> PILImage:
    """Resize image so total pixels don't exceed max_pixels, preserving aspect ratio.

    Most VLMs cap at 1MP to avoid OOM. Larger images are scaled down.
    """
    w, h = image.size
    total = w * h
    if total <= max_pixels:
        return image
    scale = (max_pixels / total) ** 0.5
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def to_base64_png(image: PILImage) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
