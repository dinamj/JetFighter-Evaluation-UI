"""Schemas for papers.csv / predictions.csv / ground_truth.csv.

Join key: atomic_id (= crop_id for single figures, `<crop_id>__p<i>` for panels).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


### LABEL VOCABULARY

IS_FIGURE_LABELS      = ("yes", "no")
CLASSIFICATION_LABELS = ("rainbow_gradient", "safe_gradient", "discrete", "other")
ACCESSIBILITY_LABELS  = ("accessible", "problematic", "n/a")


### papers.csv

PAPERS_FIELDS = [
    "pdf_id",       
    "pdf_name",      
    "pdf_path",      
    "field",         
    "is_control",   
    "source",
    "notes",
]


@dataclass(frozen=True)
class PaperRow:
    pdf_id: str
    pdf_name: str
    pdf_path: str
    field: str = ""
    is_control: str = "no"
    source: str = "local"
    notes: str = ""

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in PAPERS_FIELDS}


### predictions.csv — pipeline manifest + classifier + contrast

PREDICTION_FIELDS = [
    # identity
    "atomic_id",
    "kind",
    "crop_id",
    "panel_id",
    "pdf_id",
    "pdf_name",
    "page",
    "bbox_index",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "panel_bbox_x1", "panel_bbox_y1", "panel_bbox_x2", "panel_bbox_y2",
    "image_path",
    "dpi",
    # provenance
    "detection_confidence",
    "yolo_confidence",
    "has_caption",
    "caption_figure_number",
    "caption_text",
    "is_compound",
    "compound_rule",
    "routed_through_yolo",
    "yolo_n_panels_raw",
    "yolo_n_panels_kept",
    "figure_iou_n_merged",
    # verdict
    "is_figure",
    "classification",
    "accessibility",
    "source",
    "notes",
    "contrast_verdict",
    "contrast_score",
    "contrast_n_colors",
    "contrast_reason",
]


### ground_truth.csv

GROUND_TRUTH_FIELDS = [
    "atomic_id",
    "crop_id",
    "panel_id",
    "pdf_id",
    "pdf_name",
    "page",
    "bbox_index",
    "image_path",
    "is_figure",
    "classification",
    "accessibility",
    "duplicate_of_previous",
    "is_ishihara",
    "worst_case_panel",
    "reviewer",
    "reviewed_at",
    "notes",
]


### HELPERS

def is_known_classification(label: str) -> bool:
    return label in CLASSIFICATION_LABELS


def is_known_accessibility(label: str) -> bool:
    return label in ACCESSIBILITY_LABELS


def normalize_label(label: Optional[str]) -> str:
    if label is None:
        return ""
    return str(label).strip().lower()
