"""Paths + metric knobs for the self-contained evaluation repo.

The dataset (crops, manifest, predictions) ships inside this repo under
dataset/. Human labels are written per reviewer under results/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


EVAL_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = EVAL_ROOT / "dataset"
RESULTS_DIR = EVAL_ROOT / "results"


@dataclass
class EvalConfig:
    metrics: dict = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return DATASET_DIR / "manifest.csv"

    @property
    def predictions_path(self) -> Path:
        return DATASET_DIR / "predictions.csv"

    @property
    def ground_truth_path(self) -> Path:
        return RESULTS_DIR / "ground_truth.csv"


def load(path: Path | None = None) -> EvalConfig:
    """Load optional config.yaml; falls back to built-in metric defaults."""
    raw = {}
    if path is None:
        path = EVAL_ROOT / "config.yaml"
    if path.exists() and yaml is not None:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    return EvalConfig(metrics=raw.get("metrics", {}))
