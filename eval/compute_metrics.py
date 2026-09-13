"""Pipeline predictions vs human ground truth -> metrics.json + metrics_report.txt

  global -> pooled over all crops
  per_paper -> one confusion matrix per pdf_id
  random_page -> one random page per paper
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import EvalConfig, RESULTS_DIR
from .schemas import ACCESSIBILITY_LABELS, CLASSIFICATION_LABELS, normalize_label

FINAL_VERDICT_LABELS = ["accessible", "problematic"]


def _final_verdict(classification: str, accessibility: str) -> str | None:
    """Collapse category + accessibility into JetFighter's binary GREEN/RED output.

        accessible  <- safe_gradient,  discrete + accessible
        problematic <- rainbow_gradient, discrete + problematic

    Returns None when no verdict applies (e.g. 'other', or discrete without an
    accessibility label).
    """
    cls = normalize_label(classification)
    acc = normalize_label(accessibility)
    if cls == "safe_gradient":
        return "accessible"
    if cls == "rainbow_gradient":
        return "problematic"
    if cls == "discrete" and acc in ("accessible", "problematic"):
        return acc
    return None


def _final_pairs(df: pd.DataFrame, exclude_dupes: bool):
    """(y_true, y_pred) binary verdicts over crops the human confirmed as figures.

    A row counts whenever BOTH sides yield a verdict — even if their categories
    differ (safe_gradient vs accessible-discrete both map to 'accessible')."""
    if exclude_dupes and "duplicate_of_previous" in df.columns:
        df = df[df["duplicate_of_previous"].str.lower() != "yes"]
    valid = df[df["is_figure_gt"].str.lower() == "yes"]
    y_true, y_pred = [], []
    for _, r in valid.iterrows():
        gt = _final_verdict(r["classification_gt"], r["accessibility_gt"])
        pr = _final_verdict(r["classification_pred"], r["accessibility_pred"])
        if gt is not None and pr is not None:
            y_true.append(gt)
            y_pred.append(pr)
    return y_true, y_pred


def _final_verdict_block(df: pd.DataFrame, exclude_dupes: bool) -> dict[str, Any]:
    y_true, y_pred = _final_pairs(df, exclude_dupes)
    if not y_true:
        return {"n_evaluated": 0, "note": "no rows with a verdict on both sides"}
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=FINAL_VERDICT_LABELS, average="binary",
        pos_label="problematic", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=FINAL_VERDICT_LABELS)
    return {
        "n_evaluated": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": {"labels": FINAL_VERDICT_LABELS, "matrix": cm.astype(int).tolist()},
    }


### IO

def _merge(predictions_csv: Path, gt_csv: Path) -> pd.DataFrame:
    pred = pd.read_csv(predictions_csv, dtype=str).fillna("")
    gt = pd.read_csv(gt_csv, dtype=str).fillna("")
    pred = pred.drop_duplicates(subset="atomic_id", keep="last")
    gt = gt.drop_duplicates(subset="atomic_id", keep="last")
    return pred.merge(gt, on="atomic_id", how="inner", suffixes=("_pred", "_gt"))


### Step blocks

def _detection_block(df: pd.DataFrame, exclude_dupes: bool) -> dict[str, Any]:
    if exclude_dupes and "duplicate_of_previous" in df.columns:
        df = df[df["duplicate_of_previous"].str.lower() != "yes"]
    n = len(df)
    if n == 0:
        return {"n_proposed": 0, "n_rejected_by_human": 0, "false_positive_rate": 0.0}
    rejected = (df["is_figure_gt"].str.lower() != "yes").sum()
    return {
        "n_proposed": int(n),
        "n_rejected_by_human": int(rejected),
        "false_positive_rate": float(rejected / n),
    }


def _classification_block(df: pd.DataFrame, labels: list[str],
                          exclude_dupes: bool) -> dict[str, Any]:
    if exclude_dupes and "duplicate_of_previous" in df.columns:
        df = df[df["duplicate_of_previous"].str.lower() != "yes"]
    valid = df[df["is_figure_gt"].str.lower() == "yes"].copy()
    if valid.empty:
        return {"n_evaluated": 0, "note": "no human-confirmed figures"}

    y_true = valid["classification_gt"].map(normalize_label)
    y_pred = valid["classification_pred"].map(normalize_label)
    mask = y_true.isin(labels) & y_pred.isin(labels)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.empty:
        return {"n_evaluated": 0, "note": "no rows with valid labels"}

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n_evaluated": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "confusion_matrix": {"labels": labels, "matrix": cm.astype(int).tolist()},
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, digits=3),
    }


def _accessibility_block(df: pd.DataFrame, labels: list[str],
                         exclude_dupes: bool) -> dict[str, Any]:
    if exclude_dupes and "duplicate_of_previous" in df.columns:
        df = df[df["duplicate_of_previous"].str.lower() != "yes"]
    valid = df[
        (df["is_figure_gt"].str.lower() == "yes")
        & (df["classification_gt"].map(normalize_label) == "discrete")
        & (df["classification_pred"].map(normalize_label) == "discrete")
    ].copy()
    if valid.empty:
        return {"n_evaluated": 0, "note": "no shared discrete plots"}

    y_true = valid["accessibility_gt"].map(normalize_label)
    y_pred = valid["accessibility_pred"].map(normalize_label)
    mask = y_true.isin(labels) & y_pred.isin(labels)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.empty:
        return {"n_evaluated": 0, "note": "no rows with both labels filled"}

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="binary",
        pos_label="problematic", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n_evaluated": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": {"labels": labels, "matrix": cm.astype(int).tolist()},
    }


### Paper grouping

def _paper_key(df: pd.DataFrame) -> str:
    for c in ("pdf_id_gt", "pdf_id_pred", "pdf_id", "pdf_name_gt", "pdf_name"):
        if c in df.columns:
            return c
    raise KeyError("no pdf_id / pdf_name column")


def _page_key(df: pd.DataFrame) -> str | None:
    for c in ("page_gt", "page_pred", "page"):
        if c in df.columns:
            return c
    return None


def _cm_only(df: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
    valid = df[df["is_figure_gt"].str.lower() == "yes"].copy()
    if valid.empty:
        return {"n": 0}
    y_true = valid["classification_gt"].map(normalize_label)
    y_pred = valid["classification_pred"].map(normalize_label)
    mask = y_true.isin(labels) & y_pred.isin(labels)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.empty:
        return {"n": 0}
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix": {"labels": labels, "matrix": cm.astype(int).tolist()}}


def _per_paper_block(df: pd.DataFrame, labels: list[str], exclude_dupes: bool) -> dict[str, Any]:
    if df.empty:
        return {"n_papers": 0}
    key = _paper_key(df)
    papers = {}
    for pid, sub in df.groupby(key, sort=True):
        blk = _cm_only(sub, labels) | {"n_rows_total": int(len(sub))}
        fv = _final_verdict_block(sub, exclude_dupes)
        if "accuracy" in fv:
            blk["final_verdict_accuracy"] = fv["accuracy"]
            blk["final_verdict_n"] = fv["n_evaluated"]
        papers[str(pid)] = blk
    accs = [p["accuracy"] for p in papers.values() if p.get("n", 0) > 0]
    fv_accs = [p["final_verdict_accuracy"] for p in papers.values()
               if "final_verdict_accuracy" in p]
    return {
        "group_key": key,
        "n_papers": len(papers),
        "mean_paper_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_paper_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_paper_final_verdict_accuracy": float(np.mean(fv_accs)) if fv_accs else 0.0,
        "std_paper_final_verdict_accuracy": float(np.std(fv_accs)) if fv_accs else 0.0,
        "papers": papers,
    }


def _random_page_block(df: pd.DataFrame, cls_labels: list[str],
                       acc_labels: list[str], exclude_dupes: bool,
                       seed: int) -> dict[str, Any]:
    if df.empty:
        return {"n_papers": 0}
    paper_col = _paper_key(df)
    page_col = _page_key(df)
    if page_col is None:
        return {"note": "no page column - skipped"}

    rng = random.Random(seed)
    picked_idx, paper_to_page = [], {}
    for pid, sub in df.groupby(paper_col, sort=True):
        pages = sorted(sub[page_col].dropna().astype(str).unique())
        if not pages:
            continue
        page = rng.choice(pages)
        paper_to_page[str(pid)] = page
        picked_idx.append(sub.index[sub[page_col].astype(str) == page])
    if not picked_idx:
        return {"n_papers": 0}
    sub = df.loc[np.concatenate([i.values for i in picked_idx])]
    return {
        "n_papers": len(paper_to_page),
        "n_rows": int(len(sub)),
        "seed": seed,
        "detection": _detection_block(sub, exclude_dupes),
        "classification": _classification_block(sub, cls_labels, exclude_dupes),
        "accessibility": _accessibility_block(sub, acc_labels, exclude_dupes),
        "final_verdict": _final_verdict_block(sub, exclude_dupes),
    }


### Text report

def _fmt(metrics: dict[str, Any]) -> str:
    out = []
    def banner(t):
        out.append("\n" + "=" * 72)
        out.append(f"  {t}")
        out.append("=" * 72)

    banner(f"JetFighter Evaluation (reviewer: {metrics.get('reviewer', '?')})")
    out.append(f"  atomic units evaluated: {metrics['n_atomic_units_evaluated']}")

    g = metrics["global"]
    d, c, a, fv = g["detection"], g["classification"], g["accessibility"], g["final_verdict"]
    banner("Global")
    out.append(f"  Step1 detection FPR : {d.get('false_positive_rate', 0):.3f} "
               f"(proposed {d.get('n_proposed', 0)}, rejected {d.get('n_rejected_by_human', 0)})")
    if "accuracy" in c:
        out.append(f"  Step2 colormap acc  : {c['accuracy']:.3f}  macroF1={c['macro_f1']:.3f}  "
                   f"(n={c['n_evaluated']})")
    if "accuracy" in a:
        out.append(f"  Step3 access acc    : {a['accuracy']:.3f}  "
                   f"F1(problematic)={a['f1']:.3f}  (n={a['n_evaluated']})")
    if "accuracy" in fv:
        out.append(f"  FINAL verdict acc   : {fv['accuracy']:.3f}  "
                   f"F1(problematic)={fv['f1']:.3f}  (n={fv['n_evaluated']})  "
                   f"<- binary green/red, category-agnostic")

    pp = metrics["per_paper"]
    banner("Per-paper (independent unit = paper)")
    out.append(f"  papers={pp.get('n_papers', 0)}  "
               f"mean colormap acc={pp.get('mean_paper_accuracy', 0):.3f} "
               f"+/- {pp.get('std_paper_accuracy', 0):.3f}")
    out.append(f"  mean FINAL verdict acc={pp.get('mean_paper_final_verdict_accuracy', 0):.3f} "
               f"+/- {pp.get('std_paper_final_verdict_accuracy', 0):.3f}")

    rp = metrics["random_page"]
    if "classification" in rp and "accuracy" in rp["classification"]:
        rc, ra, rfv = rp["classification"], rp["accessibility"], rp.get("final_verdict", {})
        banner(f"Random-page (1 page/paper, seed {rp.get('seed')})")
        out.append(f"  papers={rp.get('n_papers', 0)} rows={rp.get('n_rows', 0)}  "
                   f"Step2 acc={rc['accuracy']:.3f}"
                   + (f"  Step3 acc={ra['accuracy']:.3f}" if 'accuracy' in ra else ""))
        if "accuracy" in rfv:
            out.append(f"  FINAL verdict acc={rfv['accuracy']:.3f}  "
                       f"F1(problematic)={rfv['f1']:.3f}  (n={rfv['n_evaluated']})")
    out.append("")
    return "\n".join(out)


### Public

def compute(cfg: EvalConfig, *, quiet: bool = False) -> Path:
    predictions = cfg.predictions_path
    gt = cfg.ground_truth_path
    if not predictions.exists():
        raise FileNotFoundError(f"missing {predictions}")
    if not gt.exists():
        raise FileNotFoundError(f"missing {gt} - label some crops in the UI first")

    cls_labels = list(cfg.metrics.get("classification_labels", CLASSIFICATION_LABELS))
    acc_labels = list(cfg.metrics.get("accessibility_labels",
                                      [l for l in ACCESSIBILITY_LABELS if l != "n/a"]))
    exclude_dupes = bool(cfg.metrics.get("exclude_duplicates_from_metrics", True))
    seed = int(cfg.metrics.get("random_page_seed", 42))

    merged = _merge(predictions, gt)
    reviewer = ""
    if "reviewer" in merged.columns and not merged["reviewer"].empty:
        reviewer = str(merged["reviewer"].dropna().iloc[0]) if len(merged) else ""

    metrics: dict[str, Any] = {
        "reviewer": reviewer,
        "n_atomic_units_evaluated": int(len(merged)),
        "global": {
            "detection": _detection_block(merged, exclude_dupes),
            "classification": _classification_block(merged, cls_labels, exclude_dupes),
            "accessibility": _accessibility_block(merged, acc_labels, exclude_dupes),
            "final_verdict": _final_verdict_block(merged, exclude_dupes),
        },
        "per_paper": _per_paper_block(merged, cls_labels, exclude_dupes),
        "random_page": _random_page_block(merged, cls_labels, acc_labels, exclude_dupes, seed),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cfg.metrics_report_path.write_text(_fmt(metrics), encoding="utf-8")
    if not quiet:
        print(_fmt(metrics))
        print(f"[metrics] {cfg.metrics_path}")
    return cfg.metrics_path


if __name__ == "__main__":
    from .config import load
    compute(load())
