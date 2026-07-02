"""Top-level package helpers for DGS.

The package entrypoint intentionally keeps heavyweight ML imports lazy so that
metadata lookups and CLI help stay fast in lightweight environments.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

__version__ = "0.1.0"
__author__ = "Jiaqi Li"
__email__ = "jiaqili@zju.edu.cn"


def _set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs when torch is available."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _set_torch_seed(seed: int) -> None:
    """Backward-compatible alias for setting torch-related seeds."""
    _set_random_seed(seed)


def _set_torch_backend(benchmark: bool = True) -> None:
    """Configure cuDNN determinism/benchmark flags when torch is installed."""
    try:
        import torch
    except ImportError:
        return

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(benchmark)
        torch.backends.cudnn.deterministic = not bool(benchmark)


def _get_device(gpu_id: int = 0):
    """Return the requested CUDA device when available, otherwise CPU."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("DGS requires torch for device selection.") from exc

    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def _configure_logger(output_dir: str | Path, verbose: int = 1) -> logging.Logger:
    """Create a DGS logger that writes both console and run log output."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("DGS")
    logger.setLevel(logging.DEBUG if verbose > 1 else logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(output_path / "DGS.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG if verbose > 1 else logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def initialize_logger(output_path: str | Path, verbosity: int = 2) -> logging.Logger:
    """Initialize a file/console logger for legacy tutorial notebooks.

    Older DGS/NvTK-style notebooks call ``DGS.initialize_logger(path)`` before
    building models. Keep that public helper as a thin compatibility wrapper
    while the newer CLI path continues to use ``setup_environment``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logger = logging.getLogger("DGS")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(output_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if verbosity:
        stream_formatter = logging.Formatter("%(asctime)s - %(message)s")
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    for name in ("dgs", "DGS"):
        child_logger = logging.getLogger(name)
        child_logger.setLevel(level)

    return logger


def setup_environment(
    output_dir: str | Path,
    verbose: int = 1,
    seed: int = 42,
    benchmark: bool = True,
    gpu_id: int = 0,
) -> Tuple[object, logging.Logger]:
    """Prepare the runtime environment and return ``(device, logger)``."""
    logger = _configure_logger(output_dir=output_dir, verbose=verbose)
    _set_random_seed(seed)
    _set_torch_backend(benchmark=benchmark)
    device = _get_device(gpu_id=gpu_id)
    logger.info("DGS environment initialized on %s", device)
    return device, logger


__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "initialize_logger",
    "setup_environment",
    "_get_device",
    "_set_random_seed",
    "_set_torch_backend",
    "_set_torch_seed",
]
