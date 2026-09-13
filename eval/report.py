"""Render results/report.html from metrics.json

KPIs (global / per-paper / random-page) plus a gallery of crops where the
pipeline classification disagreed with the human ground truth
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from .config import EvalConfig, RESULTS_DIR
from .schemas import normalize_label


### HELPERS

def _thumb_b64(image_abs: Path, max_dim: int = 240) -> str | None:
    if not image_abs.exists():
        return None
    img = cv2.imread(str(image_abs))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _cm_html(labels: list[str], matrix: list[list[int]]) -> str:
    head = "<tr><th>GT \\ Pred</th>" + "".join(f"<th>{l}</th>" for l in labels) + "</tr>"
    rows = []
    for i, l in enumerate(labels):
        cells = "".join(
            f'<td class="{"diag" if i == j else "off"}">{matrix[i][j]}</td>'
            for j in range(len(labels)))
        rows.append(f"<tr><th>{l}</th>{cells}</tr>")
    return f'<table class="cm">{head}{"".join(rows)}</table>'


_STYLES = """
body{font-family:-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}
h1{border-bottom:2px solid #444}h2{margin-top:2em;color:#1a4d8c;border-bottom:1px solid #ddd}
table.cm th,table.cm td{border:1px solid #bbb;padding:6px 10px;text-align:right}
table.cm td.diag{background:#e6f3ff;font-weight:bold}table.cm td.off{background:#fff5f5}
.kpi{display:inline-block;margin:0 1em .5em 0;padding:.5em 1em;background:#f3f3f3;border-left:4px solid #1a4d8c}
.kpi b{display:block;font-size:1.3em}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:.75em}
.card{border:1px solid #ccc;border-radius:4px;overflow:hidden;font-size:.85em}
.card img{width:100%;display:block;background:#f0f0f0}.card .meta{padding:.4em .6em}
.gt{color:#1a8c1a}.pred{color:#c0392b}
"""


def _kpi(label: str, value: str) -> str:
    return f'<div class="kpi">{label}<b>{value}</b></div>'


def _gallery(cfg: EvalConfig, cap: int = 24) -> list[dict[str, Any]]:
    pred = pd.read_csv(cfg.predictions_path, dtype=str).fillna("")
    gt = pd.read_csv(cfg.ground_truth_path, dtype=str).fillna("")
    man = pd.read_csv(cfg.manifest_path, dtype=str).fillna("")
    m = pred.merge(gt, on="atomic_id", how="inner", suffixes=("_pred", "_gt"))
    m = m.merge(man[["atomic_id", "image_path"]], on="atomic_id", how="left")
    m["cp"] = m["classification_pred"].map(normalize_label)
    m["cg"] = m["classification_gt"].map(normalize_label)
    bad = m[(m["is_figure_gt"].str.lower() == "yes") & (m["cp"] != m["cg"])]
    return [{"atomic_id": r["atomic_id"], "gt": r["cg"], "pred": r["cp"],
             "image_path": r.get("image_path", "")}
            for _, r in bad.head(cap).iterrows()]


### PUBLIC

def generate(cfg: EvalConfig, *, quiet: bool = False) -> dict[str, Path]:
    metrics_json = cfg.metrics_path
    if not metrics_json.exists():
        raise FileNotFoundError(f"missing {metrics_json.name} - run `evaluation metrics` first.")
    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    g = metrics["global"]
    d, c, a = g["detection"], g["classification"], g["accessibility"]
    fv = g.get("final_verdict", {})
    pp, rp = metrics["per_paper"], metrics["random_page"]

    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>JetFighter Evaluation</title><style>{_STYLES}</style></head><body>",
             f"<h1>JetFighter Evaluation</h1>",
             f"<p>Reviewer: <b>{metrics.get('reviewer', '?')}</b> &middot; "
             f"{metrics['n_atomic_units_evaluated']} atomic units</p>",
             "<h2>Global</h2>",
             _kpi("Detection FPR", f"{d.get('false_positive_rate', 0):.3f}")]
    if "accuracy" in c:
        parts.append(_kpi("Colormap acc", f"{c['accuracy']:.3f}"))
        parts.append(_kpi("Colormap macro-F1", f"{c['macro_f1']:.3f}"))
    if "accuracy" in a:
        parts.append(_kpi("Access acc", f"{a['accuracy']:.3f}"))
        parts.append(_kpi("Access F1 (problematic)", f"{a['f1']:.3f}"))
    if "confusion_matrix" in c:
        parts.append("<h3>Colormap confusion matrix</h3>")
        parts.append(_cm_html(c["confusion_matrix"]["labels"], c["confusion_matrix"]["matrix"]))

    if "accuracy" in fv:
        parts.append("<h2>Final verdict (binary green/red, category-agnostic)</h2>")
        parts.append("<p>The end output: accessible (safe gradient or accessible "
                     "discrete) vs problematic (rainbow or problematic discrete). "
                     "Counts as correct whenever both sides land on the same verdict, "
                     "even if the category label differs.</p>")
        parts.append(_kpi("Accuracy", f"{fv['accuracy']:.3f}"))
        parts.append(_kpi("F1 (problematic)", f"{fv['f1']:.3f}"))
        parts.append(_kpi("Precision", f"{fv['precision']:.3f}"))
        parts.append(_kpi("Recall", f"{fv['recall']:.3f}"))
        parts.append(_cm_html(fv["confusion_matrix"]["labels"], fv["confusion_matrix"]["matrix"]))

    parts.append("<h2>Per-paper (independent unit = paper)</h2>")
    parts.append(_kpi("Papers", str(pp.get("n_papers", 0))))
    parts.append(_kpi("Mean colormap acc",
                      f"{pp.get('mean_paper_accuracy', 0):.3f} +/- {pp.get('std_paper_accuracy', 0):.3f}"))
    parts.append(_kpi("Mean final-verdict acc",
                      f"{pp.get('mean_paper_final_verdict_accuracy', 0):.3f} +/- "
                      f"{pp.get('std_paper_final_verdict_accuracy', 0):.3f}"))

    if "classification" in rp and "accuracy" in rp["classification"]:
        parts.append("<h2>Random-page (1 page/paper)</h2>")
        parts.append(_kpi("Papers", str(rp.get("n_papers", 0))))
        parts.append(_kpi("Colormap acc", f"{rp['classification']['accuracy']:.3f}"))
        if "accuracy" in rp["accessibility"]:
            parts.append(_kpi("Access acc", f"{rp['accessibility']['accuracy']:.3f}"))
        rfv = rp.get("final_verdict", {})
        if "accuracy" in rfv:
            parts.append(_kpi("Final-verdict acc", f"{rfv['accuracy']:.3f}"))

    examples = _gallery(cfg)
    if examples:
        parts.append("<h2>Confusing crops</h2><div class='gallery'>")
        for ex in examples:
            b64 = _thumb_b64(cfg.dataset_dir / ex["image_path"]) if ex["image_path"] else None
            img = (f'<img src="data:image/jpeg;base64,{b64}"/>' if b64
                   else '<div style="height:120px;background:#eee"></div>')
            parts.append(f"<div class='card'>{img}<div class='meta'>"
                         f"<code>{ex['atomic_id']}</code><br>"
                         f"<span class='gt'>GT: {ex['gt']}</span><br>"
                         f"<span class='pred'>Pred: {ex['pred']}</span></div></div>")
        parts.append("</div>")
    parts.append("</body></html>")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = cfg.report_path
    html_path.write_text("".join(parts), encoding="utf-8")
    if not quiet:
        print(f"[report] {html_path}")
    return {"html": html_path, "json": metrics_json}


if __name__ == "__main__":
    from .config import load
    generate(load())
