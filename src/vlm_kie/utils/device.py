"""GPU/CPU device detection utilities."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("CUDA device: %s (%.1f GB VRAM)", name, vram_gb)
            return "cuda"
    except ImportError:
        pass
    logger.info("No CUDA device found — using CPU.")
    return "cpu"


def get_free_vram_gb() -> float:
    """Return free VRAM in GB, or 0.0 if no CUDA device."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return free / 1e9
    except ImportError:
        pass
    return 0.0
