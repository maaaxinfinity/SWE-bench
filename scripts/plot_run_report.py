#!/usr/bin/env python3
"""
Plot SWE-bench run reports produced by the official harness.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


def load_report(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_label(path: Path) -> str:
    stem = path.stem
    match = re.search(r"-run(\d+)$", stem)
    if match:
        run_idx = match.group(1)
        stem = stem[: match.start()]
        return f"{stem}@{run_idx}"
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SWE-bench run report JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        type=str,
        required=True,
        help="Report JSON files (e.g. minimax-nvfp4.run_id.json)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots",
        help="Output directory for plots and CSV summary",
    )
    parser.add_argument(
        "--rate_denominator",
        type=str,
        choices=["total", "submitted"],
        default="total",
        help="Denominator for resolution rate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for report_path in args.reports:
        path = Path(report_path)
        data = load_report(path)
        total = data.get("total_instances", 0)
        submitted = data.get("submitted_instances", 0)
        resolved = data.get("resolved_instances", 0)
        unresolved = data.get("unresolved_instances", 0)
        errors = data.get("error_instances", 0)
        empty = data.get("empty_patch_instances", 0)
        denom = total if args.rate_denominator == "total" else submitted
        rate = resolved / denom if denom else 0.0
        rows.append(
            {
                "report": path.name,
                "total": total,
                "submitted": submitted,
                "resolved": resolved,
                "unresolved": unresolved,
                "errors": errors,
                "empty": empty,
                "resolution_rate": rate,
            }
        )

    csv_path = out_dir / "run_report_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib not available, wrote CSV only: {exc}")
        return

    labels = [format_label(Path(row["report"])) for row in rows]
    rates = [row["resolution_rate"] * 100 for row in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4))
    ax.bar(labels, rates, color="#2f6f9f")
    ax.set_ylabel("Resolution Rate (%)")
    ax.set_title("SWE-bench Resolution Rate")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()

    plot_path = out_dir / "resolution_rate.png"
    fig.savefig(plot_path, dpi=200)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
