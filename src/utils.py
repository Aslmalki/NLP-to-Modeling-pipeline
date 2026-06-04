"""
Shared utilities. Step 0 of pipeline.
See config.py for all parameter values.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from typing import Optional

# Default seed matches config.REPRODUCIBILITY_SEED
_DEFAULT_SEED = 42


def setup_logging(
    level: int = logging.INFO,
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure a module logger with a consistent stream handler.
    Used across pipeline modules for reproducible, readable run logs.
    """
    logger_name = name or "nlp_to_abm_pipeline"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_all_seeds(seed: int = _DEFAULT_SEED) -> None:
    """
    Set random seeds for reproducible results (ODE 1.3.3).

    Covers Python ``random``, NumPy, PyTorch (CPU/CUDA), and UMAP
    (via NumPy's global RNG and ``UMAP_RANDOM_STATE`` when applicable).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # UMAP uses NumPy random state; document expected random_state for fits
    os.environ.setdefault("UMAP_RANDOM_STATE", str(seed))
