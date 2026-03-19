#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = ROOT / "runs" / "detect" / "train" / "results.csv"

DISPLAY_COLUMNS = [
    ("epoch", "epoch"),
    ("time", "time_s"),
    ("train/box_loss", "tr_box"),
    ("train/cls_loss", "tr_cls"),
    ("train/dfl_loss", "tr_dfl"),
    ("metrics/precision(B)", "prec"),
    ("metrics/recall(B)", "rec"),
    ("metrics/mAP50(B)", "map50"),
    ("metrics/mAP50-95(B)", "map5095"),
    ("val/box_loss", "val_box"),
    ("val/cls_loss", "val_cls"),
    ("val/dfl_loss", "val_dfl"),
    ("lr/pg0", "lr"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretty-print a YOLO results.csv file in a readable table."
    )
    parser.add_argument(
        "results_path",
        nargs="?",
        default=str(DEFAULT_RESULTS_PATH),
        help="Path to the results.csv file.",
    )
    return parser.parse_args()


def format_value(value: str, key: str) -> str:
    if key == "epoch":
        return str(int(float(value)))
    if key == "time":
        return f"{float(value):.1f}"
    if key == "lr/pg0":
        return f"{float(value):.2e}"
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_path).expanduser().resolve()

    if not results_path.exists():
        raise SystemExit(f"Missing file: {results_path}")

    with results_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise SystemExit(f"No rows found in: {results_path}")

    headers = [label for _, label in DISPLAY_COLUMNS]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            [format_value(row[source_key], source_key) for source_key, _ in DISPLAY_COLUMNS]
        )

    widths = []
    for index, header in enumerate(headers):
        cell_width = max(len(header), *(len(row[index]) for row in formatted_rows))
        widths.append(cell_width)

    print(f"Results: {results_path}")
    print()
    print("  ".join(header.rjust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(widths))))
    for row in formatted_rows:
        print("  ".join(row[i].rjust(widths[i]) for i in range(len(row))))

    best_row = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))
    print()
    print("Best epoch by mAP50")
    print(f"- epoch: {int(float(best_row['epoch']))}")
    print(f"- mAP50: {float(best_row['metrics/mAP50(B)']):.4f}")
    print(f"- precision: {float(best_row['metrics/precision(B)']):.4f}")
    print(f"- recall: {float(best_row['metrics/recall(B)']):.4f}")
    print(f"- val_box_loss: {float(best_row['val/box_loss']):.4f}")
    print(f"- learning_rate: {float(best_row['lr/pg0']):.2e}")


if __name__ == "__main__":
    main()
