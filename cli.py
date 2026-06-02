"""evaluation CLI (ui / metrics / report)"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .eval import compute_metrics, config, report


def cmd_ui(port: int) -> int:
    ui_app = Path(__file__).parent / "eval" / "ui_app.py"
    argv = [sys.executable, "-m", "streamlit", "run", str(ui_app),
            "--server.port", str(port), "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--", "--eval-root", str(config.EVAL_ROOT)]
    print(f"[ui] http://localhost:{port}")
    return subprocess.call(argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluation")
    sub = p.add_subparsers(dest="command", required=True)

    p_ui = sub.add_parser("ui", help="Launch the labelling UI")
    p_ui.add_argument("--port", type=int, default=8501)

    sub.add_parser("metrics", help="Compute pipeline-vs-human metrics")
    sub.add_parser("report", help="Render results/report.html")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load()

    if args.command == "ui":
        return cmd_ui(args.port)
    if args.command == "metrics":
        compute_metrics.compute(cfg)
        return 0
    if args.command == "report":
        report.generate(cfg)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
