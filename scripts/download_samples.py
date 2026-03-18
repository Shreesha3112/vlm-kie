#!/usr/bin/env python
"""Download CORD-v2 test split samples from HuggingFace datasets.

Usage:
    uv run scripts/download_samples.py [--n 100]
"""

import argparse
import sys
from pathlib import Path

# Ensure src is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlm_kie.data.loader import load_cord_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CORD-v2 sample images.")
    parser.add_argument("--n", type=int, default=100, help="Number of samples (default: 100)")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    args = parser.parse_args()

    samples = load_cord_v2(n=args.n, split=args.split)
    print(f"Downloaded {len(samples)} samples.")
    print(f"Saved to: {samples[0]['image_path'].parent}")


if __name__ == "__main__":
    main()
