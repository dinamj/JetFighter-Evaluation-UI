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
    dataset: str = "dataset"
    metrics: dict = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return self.dataset_dir / "manifest.csv"

    @property
    def predictions_path(self) -> Path:
        return self.dataset_dir / "predictions.csv"

    @property
    def dataset_dir(self) -> Path:
        return EVAL_ROOT / self.dataset

    @property
    def result_suffix(self) -> str:
        return "" if self.dataset == "dataset" else f"_{Path(self.dataset).name}"

    @property
    def ground_truth_path(self) -> Path:
        return RESULTS_DIR / f"ground_truth{self.result_suffix}.csv"

    @property
    def metrics_path(self) -> Path:
        return RESULTS_DIR / f"metrics{self.result_suffix}.json"

    @property
    def metrics_report_path(self) -> Path:
        return RESULTS_DIR / f"metrics_report{self.result_suffix}.txt"

    @property
    def report_path(self) -> Path:
        return RESULTS_DIR / f"report{self.result_suffix}.html"


def load(path: Path | None = None, dataset: str = "dataset") -> EvalConfig:
    """Load optional config.yaml; falls back to built-in metric defaults."""
    raw = {}
    if path is None:
        path = EVAL_ROOT / "config.yaml"
    if path.exists() and yaml is not None:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    return EvalConfig(dataset=dataset, metrics=raw.get("metrics", {}))
