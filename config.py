"""
Configuration for NLP-to-ABM pipeline — Human Behavior in Extreme Environments.
ODE-compliant: paths, seeds, and constants for reproducibility.

Layout: repository root contains ``config.py``, ``data/``, ``outputs/``, ``src/``.
"""
import os
import random
from typing import Optional

# --- PyTorch version check (required for transformers model loading) ---
MIN_TORCH_VERSION = (2, 6)


def check_torch_version():
    """
    Verify PyTorch is >= 2.6 for safe model loading (CVE-2025-32434).
    If torch < 2.6, we rely on use_safetensors=True when loading BERT models.
    Raises SystemExit only if torch is not installed.
    """
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "PyTorch is not installed. Install with: pip install torch>=2.6.0"
        )
    version = getattr(torch, "__version__", "0.0.0")
    try:
        parts = tuple(int(x) for x in version.split("+")[0].split(".")[:2])
    except (ValueError, AttributeError):
        parts = (0, 0)
    if parts < MIN_TORCH_VERSION:
        print(
            f"Warning: PyTorch {version} is below 2.6. BERT models will load with safetensors only. "
            f"For full compatibility, upgrade: pip install torch>=2.6.0"
        )


# --- Reproducibility (ODE 1.3.3) ---
REPRODUCIBILITY_SEED = 42


def set_all_seeds(seed: int = REPRODUCIBILITY_SEED) -> None:
    """Set random seeds for reproducible results."""
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
    os.environ.setdefault("UMAP_RANDOM_STATE", str(seed))

# --- Paths (repository root = directory containing this file) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FOLDER = os.path.join(DATA_DIR, "corpus")
CODEBOOK_DIR = os.path.join(DATA_DIR, "codebook")
CODEBOOK_CSV = os.path.join(CODEBOOK_DIR, "topic_definitions.csv")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_FOLDER, "figures")
TOPIC_ASSIGNMENTS_DIR = os.path.join(OUTPUT_FOLDER, "topic_assignments")
VALIDATION_DIR = os.path.join(OUTPUT_FOLDER, "validation")
HUMAN_ANNOTATION_CSV = os.path.join(
    ANNOTATIONS_DIR, "HumanBehaviorInExtremeEnvironments.csv"
)

# --- Model parameters (from ablation) ---
MIN_SIZE_RANGE = range(2, 11, 2)  # [2, 4, 6, 8, 10]
MAX_VOCAB_WORDS = 1000  # For co-occurrence networks
TOP_EDGES_FOR_VIZ = 75  # Fewer edges for readable co-occurrence network figures

# Topic labels for interpretation (Topic 2 = Specialized Remote Sensing and Imaging Hardware)
TOPIC_LABELS = {
    -1: "Outliers (unassigned documents)",
    0: "Climate, Environment & Health Research",
    1: "Team, Agent & Organizational Performance",
    2: "Specialized Remote Sensing and Imaging Hardware",
}
DOC2VEC_SEED = REPRODUCIBILITY_SEED
UMAP_RANDOM_STATE = REPRODUCIBILITY_SEED

# Fixed validation subset for Cohen's kappa (loaded at runtime; do not derive dynamically).
VALIDATION_SUBSET_IDS_PATH = os.path.join(DATA_DIR, "validation_subset_ids.txt")
EXPECTED_VALIDATION_SUBSET_N = 63

# LLM validation (run_blue_ocean_pipeline.py) — single-model for reproducibility.
LLM_TEMPERATURE = 0
LLM_SEED = 42  # Reserved for optional Gemini path only; not used in reported Claude runs.
# Single-model classification for reproducibility.
# Gemini fallback available via ENABLE_GEMINI_FALLBACK but not used in reported results.
ENABLE_GEMINI_FALLBACK = False
LLM_MODEL = "claude-sonnet-4-6"


def load_validation_subset_ids(path: Optional[str] = None) -> frozenset:
    """Load the pinned list of Paper IDs used for kappa / confusion-matrix reporting."""
    path = path or VALIDATION_SUBSET_IDS_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Validation subset file not found: {path}. "
            "Create data/validation_subset_ids.txt with exactly 63 Paper IDs."
        )
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    if len(ids) != EXPECTED_VALIDATION_SUBSET_N:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_SUBSET_N} validation Paper IDs in {path}, got {len(ids)}."
        )
    return frozenset(ids)
