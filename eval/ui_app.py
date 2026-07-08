"""Streamlit UI

Three-step protocol (Detection -> Colormap -> Accessibility) with side-by-side
color + BT.709 grayscale view. Labels are written to results/ground_truth.csv;
progress auto-saves per row, so closing and resuming later is safe

Run:
    python -m evaluation ui
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

EVAL_ROOT = Path(__file__).resolve().parent.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from eval.schemas import GROUND_TRUTH_FIELDS  # noqa: E402

# Inlined verbatim from inference/contrast_analysis.py so this folder ships
# without any dependency on the surrounding JetFighter repo.
def _grayscale_bt709(bgr) -> float:
    return 0.0722 * bgr[0] + 0.7152 * bgr[1] + 0.2126 * bgr[2]


### CONSTANTS

CLASSIFICATION_CHOICES = [
    ("Rainbow Gradient",  "rainbow_gradient", "R"),
    ("Safe Gradient",     "safe_gradient",    "S"),
    ("Discrete Plot",     "discrete",         "C"),
]

ACCESSIBILITY_CHOICES = [
    ("Accessible",   "accessible",   "A"),
    ("Problematic",  "problematic",  "P"),
]

# Global action keys
KEY_DUPLICATE = "D"
KEY_SUBMIT    = "↵"   # Enter
KEY_PREVIOUS  = "←"   # ArrowLeft
KEY_SKIP      = "→"   # ArrowRight

# Step 1 decision keys
KEY_YES = "Y"
KEY_NO  = "N"


### KEYBOARD SHORTCUTS

_KEYBOARD_JS = """
<script>
(function() {
    const parent = window.parent;
    const doc = parent.document;

    if (parent.__jf_kb_handler) {
        doc.removeEventListener('keydown', parent.__jf_kb_handler);
    }

    function clickButtonWithHint(hint) {
        const target = '(' + hint + ')';
        const buttons = doc.querySelectorAll('button');
        for (const btn of buttons) {
            if (btn.disabled) continue;
            if (btn.offsetParent === null) continue;
            if (btn.textContent && btn.textContent.indexOf(target) !== -1) {
                btn.click();
                return true;
            }
        }
        return false;
    }

    const LETTER_KEYS = {
        'y': 'Y', 'n': 'N',
        'r': 'R', 's': 'S', 'c': 'C',
        'a': 'A', 'p': 'P',
        'd': 'D'
    };

    parent.__jf_kb_handler = function(e) {
        const tag = (e.target && e.target.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea') return;
        if (e.target && e.target.isContentEditable) return;
        if (e.ctrlKey || e.metaKey || e.altKey) return;

        let hint = null;
        if (e.key === 'Enter')      hint = '↵';
        else if (e.key === 'ArrowLeft')  hint = '←';
        else if (e.key === 'ArrowRight') hint = '→';
        else {
            const k = (e.key || '').toLowerCase();
            if (LETTER_KEYS[k]) hint = LETTER_KEYS[k];
        }

        if (hint && clickButtonWithHint(hint)) {
            e.preventDefault();
            e.stopPropagation();
        }
    };

    doc.addEventListener('keydown', parent.__jf_kb_handler);
})();
</script>
"""


def inject_shortcuts() -> None:
    """Inject the keyboard-shortcut listener (idempotent across reruns)."""
    components.html(_KEYBOARD_JS, height=0)


def render_shortcuts_legend() -> None:
    """Sidebar legend — must stay in sync with `_KEYBOARD_JS`."""
    st.markdown(
        f"""
**Step 1 — Detection**
- `{KEY_YES}`  Yes — valid figure
- `{KEY_NO}`  No — not a figure

**Step 2 — Classification**
- `{CLASSIFICATION_CHOICES[0][2]}`  Rainbow Gradient
- `{CLASSIFICATION_CHOICES[1][2]}`  Safe Gradient
- `{CLASSIFICATION_CHOICES[2][2]}`  Discrete / Categorical

**Step 3 — Accessibility**
- `{ACCESSIBILITY_CHOICES[0][2]}`  Accessible
- `{ACCESSIBILITY_CHOICES[1][2]}`  Problematic

**Navigation (any step)**
- `{KEY_DUPLICATE}`   Duplicate of previous
- `Enter`  Submit & next
- `←`  Previous
- `→`  Skip
""",
    )


### HELPERS

def grayscale_simulation(crop_bgr: np.ndarray) -> np.ndarray:
    """BT.709 grayscale via the canonical pipeline function."""
    bgr_chw = crop_bgr.transpose(2, 0, 1).astype(np.float64)
    gray = _grayscale_bt709(bgr_chw)
    return np.clip(gray, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--dataset", default="dataset",
                        help="Dataset folder under eval-root (default: dataset). "
                             "e.g. --dataset ishihara_dataset for the Ishihara set.")
    return parser.parse_args(argv)


@st.cache_data(show_spinner=False)
def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_completed_lookup(results_csv: Path) -> dict[str, dict]:
    """Return {atomic_id: most-recent ground-truth row}."""
    if not results_csv.exists():
        return {}
    try:
        df = pd.read_csv(results_csv, dtype=str).fillna("")
        df = df.drop_duplicates(subset="atomic_id", keep="last")
        return {r["atomic_id"]: r.to_dict() for _, r in df.iterrows()}
    except Exception:
        return {}


def append_row(results_csv: Path, row: dict) -> None:
    new_file = not results_csv.exists()
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _row_template(manifest_row: pd.Series) -> dict:
    return {
        "atomic_id":              str(manifest_row.get("atomic_id", "")),
        "crop_id":                str(manifest_row.get("crop_id", "")),
        "panel_id":               str(manifest_row.get("panel_id", "") or ""),
        "pdf_id":                 str(manifest_row.get("pdf_id", "") or ""),
        "pdf_name":               str(manifest_row.get("pdf_name", "")),
        "page":                   str(manifest_row.get("page", "")),
        "bbox_index":             str(manifest_row.get("bbox_index", "")),
        "image_path":             str(manifest_row.get("image_path", "")),
        "is_figure":              "",
        "classification":         "",
        "accessibility":          "",
        "duplicate_of_previous":  "no",
        "is_ishihara":            "no",
        "worst_case_panel":       "no",
        "reviewer":               "",
        "reviewed_at":            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes":                  "",
    }


### STATE

DEFAULTS = {
    "step":            1,
    "is_figure":       None,
    "classification":  None,
    "accessibility":   None,
}


def reset_step_state() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def goto_idx(new_idx: int, n_total: int) -> None:
    st.session_state.idx = max(0, min(n_total - 1, new_idx))
    reset_step_state()


def first_pending(df: pd.DataFrame, completed: dict[str, dict]) -> int:
    for i, atomic_id in enumerate(df["atomic_id"].astype(str)):
        if atomic_id not in completed:
            return i
    return 0


### MAIN

def main() -> None:
    args = parse_args()
    dataset_dir = (args.eval_root / args.dataset).resolve()
    manifest_csv = dataset_dir / "manifest.csv"
    # A non-default dataset writes its own ground-truth file so it never touches
    # the main results/ground_truth.csv.
    gt_name = ("ground_truth.csv" if args.dataset == "dataset"
               else f"ground_truth_{Path(args.dataset).name}.csv")
    results_csv = (args.eval_root / "results" / gt_name).resolve()

    st.set_page_config(page_title="JetFighter Validation", layout="wide")
    st.title("JetFighter Evaluation")

    inject_shortcuts()

    if not manifest_csv.exists():
        st.error(f"Manifest not found at {manifest_csv}\n\n"
                 "Run `python -m evaluation build-manifest` first.")
        return

    df = load_manifest(manifest_csv).reset_index(drop=True)
    if df.empty:
        st.warning("Manifest is empty — nothing to validate.")
        return

    completed = load_completed_lookup(results_csv)

    if "idx" not in st.session_state:
        st.session_state.idx = first_pending(df, completed)
        reset_step_state()

    ### Sidebar
    n_total = len(df)
    n_done = len(completed)
    with st.sidebar:
        st.header("Progress")
        st.progress(n_done / max(n_total, 1), text=f"{n_done} / {n_total} validated")
        st.success("Auto-saved on every submit. Close the tab anytime and "
                   "reopen to resume on the next unlabelled crop.")
        st.code(str(results_csv), language="text")

        st.divider()
        st.subheader("Navigate")
        jump = st.number_input("Jump to #", min_value=1, max_value=n_total,
                               value=st.session_state.idx + 1, step=1)
        if jump - 1 != st.session_state.idx:
            goto_idx(int(jump) - 1, n_total)

        skip_done = st.checkbox("Skip already-validated", value=True)
        if skip_done and df.iloc[st.session_state.idx]["atomic_id"] in completed:
            nxt = first_pending(df.iloc[st.session_state.idx + 1:], completed)
            if nxt > 0 or df.iloc[st.session_state.idx + 1:].iloc[0:1].shape[0]:
                goto_idx(st.session_state.idx + 1 + nxt, n_total)

        st.divider()
        st.markdown("**Protocol**\n\n"
                    "1. Is the crop a scientific figure?\n"
                    "2. What colormap does it use?\n"
                    "3. (Discrete only) Distinguishable in grayscale?")

        st.divider()
        with st.expander("⌨ Keyboard shortcuts", expanded=True):
            render_shortcuts_legend()

    ### Current row
    row = df.iloc[st.session_state.idx]
    atomic_id = str(row["atomic_id"])
    image_abs = (dataset_dir / str(row["image_path"])).resolve()

    st.markdown(
        f"**PDF:** {row['pdf_name']} &nbsp;|&nbsp; "
        f"**Page:** {row['page']} &nbsp;|&nbsp; "
        f"**Atomic:** `{atomic_id}` &nbsp;|&nbsp; "
        f"**Kind:** {row.get('kind', 'crop')} &nbsp;|&nbsp; "
        f"**Index:** {st.session_state.idx + 1} / {n_total}"
    )

    if atomic_id in completed:
        prior = completed[atomic_id]
        st.info(f"Already validated as **{prior.get('classification', '?')}** / "
                f"**{prior.get('accessibility', '?')}**. Submitting again "
                f"overwrites on next deduplication.")

    if not image_abs.exists():
        st.error(f"Image not found: {image_abs}")
        return

    crop_bgr = cv2.imread(str(image_abs))
    if crop_bgr is None:
        st.error(f"Could not read image: {image_abs}")
        return
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    gray = grayscale_simulation(crop_bgr)

    col_l, col_r = st.columns(2)
    with col_l:
        st.caption("Original (left panel)")
        st.image(crop_rgb, use_column_width=True)
    with col_r:
        st.caption("Grayscale simulation — BT.709 (right panel)")
        st.image(gray, use_column_width=True, clamp=True)

    ### Duplicate-of-Previous shortcut
    can_duplicate = (
        st.session_state.idx > 0
        and str(df.iloc[st.session_state.idx - 1]["atomic_id"]) in completed
    )
    if can_duplicate:
        if st.button(f"📑 Duplicate of Previous ({KEY_DUPLICATE})", use_container_width=True):
            prev_atomic = str(df.iloc[st.session_state.idx - 1]["atomic_id"])
            prev = completed[prev_atomic]
            out = _row_template(row)
            out.update({
                "is_figure":              prev.get("is_figure", ""),
                "classification":         prev.get("classification", ""),
                "accessibility":          prev.get("accessibility", ""),
                "duplicate_of_previous":  "yes",
            })
            append_row(results_csv, out)
            completed[atomic_id] = out
            goto_idx(st.session_state.idx + 1, n_total)
            st.rerun()

    st.divider()

    ### STEP 1 — DETECTION
    st.subheader("Step 1 — Detection")
    with st.expander("ℹ️ When to mark Yes / No", expanded=False):
        st.markdown(
            "_Did the crop capture a valid scientific figure?_\n\n"
            "- **Yes** → plot, chart, heatmap, diagram, scientific image, etc.\n"
            "- **No** → isolated text blocks, tables, publisher logos, "
            "standalone formulas, or blank."
        )
    c1, c2 = st.columns(2)
    if c1.button(f"✅ Yes — valid figure ({KEY_YES})", use_container_width=True,
                 type=("primary" if st.session_state.is_figure == "yes" else "secondary")):
        st.session_state.is_figure = "yes"
        st.session_state.step = 2
    if c2.button(f"❌ No — text / table / logo / blank ({KEY_NO})",
                 use_container_width=True,
                 type=("primary" if st.session_state.is_figure == "no" else "secondary")):
        st.session_state.is_figure = "no"
        st.session_state.classification = "other"
        st.session_state.accessibility = "n/a"
        st.session_state.step = 99

    ### STEP 2 — CLASSIFICATION
    if st.session_state.is_figure == "yes":
        st.subheader("Step 2 — Colormap")
        with st.expander("ℹ️ Rainbow / Safe / Discrete — definitions", expanded=False):
            st.markdown(
                "_Which colormap does the original (left panel) use?_\n\n"
                "**Rainbow Gradient** — purple→blue→cyan→green→yellow→orange→red. "
                "Brightest colors sit in the middle; no monotonic luminance ramp.\n\n"
                "**Safe Gradient** — brightest color at one end (usually yellow), "
                "monotonically darker toward the other (e.g. viridis, plasma).\n\n"
                "**Discrete Plot** — distinct categorical colors for separate "
                "groups: bar/line/scatter plots, schemes, B&W figures, photos, "
                "microscopy."
            )
        cols = st.columns(len(CLASSIFICATION_CHOICES))
        for col, (label, value, hint) in zip(cols, CLASSIFICATION_CHOICES):
            active = st.session_state.classification == value
            if col.button(f"{label} ({hint})", use_container_width=True,
                          key=f"cls_{value}",
                          type=("primary" if active else "secondary")):
                st.session_state.classification = value
                if value == "discrete":
                    st.session_state.step = 3
                else:
                    st.session_state.accessibility = "n/a"
                    st.session_state.step = 99

    ### STEP 3 — ACCESSIBILITY
    if (st.session_state.is_figure == "yes"
            and st.session_state.classification == "discrete"):
        st.subheader("Step 3 — Accessibility")
        with st.expander("ℹ️ Accessible / Problematic — and how to read legends",
                         expanded=False):
            st.markdown(
                "_Compare original (left) to grayscale (right)._\n\n"
                "- **Accessible** → categories stay readable in grayscale.\n"
                "- **Problematic** → categories collapse into similar gray "
                "tones; if you have to strain, label Problematic.\n\n"
                "**Legends matter even if plot bars don't touch.** Two colors "
                "that look fine far apart in the plot still have to be "
                "distinguishable side-by-side in the legend. Judge the worst "
                "of the two views: plot contrast and legend contrast."
            )
        a1, a2 = st.columns(2)
        for col, (label, value, hint) in zip((a1, a2), ACCESSIBILITY_CHOICES):
            active = st.session_state.accessibility == value
            icon = "🟢" if value == "accessible" else "🔴"
            if col.button(f"{icon} {label} ({hint})", use_container_width=True,
                          type=("primary" if active else "secondary"),
                          key=f"acc_{value}"):
                st.session_state.accessibility = value
                st.session_state.step = 99

    ### Submit / nav
    st.divider()

    ready = (
        st.session_state.is_figure is not None
        and (st.session_state.is_figure == "no"
             or (st.session_state.classification is not None
                 and (st.session_state.classification != "discrete"
                      or st.session_state.accessibility is not None)))
    )

    nav_l, nav_mid, nav_r = st.columns([1, 2, 1])
    if nav_l.button(f"⟵ Previous ({KEY_PREVIOUS})", use_container_width=True):
        goto_idx(st.session_state.idx - 1, n_total)
        st.rerun()
    if nav_mid.button(f"✔ Submit & next ⟶ ({KEY_SUBMIT})", type="primary",
                      disabled=not ready, use_container_width=True):
        out = _row_template(row)
        out.update({
            "is_figure":      st.session_state.is_figure or "",
            "classification": st.session_state.classification or "other",
            "accessibility":  st.session_state.accessibility or "n/a",
        })
        append_row(results_csv, out)
        completed[atomic_id] = out
        goto_idx(st.session_state.idx + 1, n_total)
        st.rerun()
    if nav_r.button(f"Skip ⟶ ({KEY_SKIP})", use_container_width=True):
        goto_idx(st.session_state.idx + 1, n_total)
        st.rerun()


if __name__ == "__main__":
    main()
