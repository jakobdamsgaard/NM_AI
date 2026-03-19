#!/usr/bin/env python3

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = ROOT / "annotations.json"
IMAGES_DIR = ROOT / "images"
OUTPUT_DIR = ROOT / "exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a smaller YOLO dataset from a COCO-style annotations file."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of most common classes to keep.",
    )
    parser.add_argument(
        "--min-boxes",
        type=int,
        default=1,
        help="Only keep classes with at least this many boxes.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Keep the 'unknown_product' class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/val split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of images to use for training.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional custom name for the export folder.",
    )
    return parser.parse_args()


def yolo_box(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    norm_w = w / width
    norm_h = h / height
    return x_center, y_center, norm_w, norm_h


def main() -> None:
    args = parse_args()

    with ANNOTATIONS_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    category_names = {category["id"]: category["name"] for category in categories}
    image_lookup = {image["id"]: image for image in images}

    category_counts = Counter(annotation["category_id"] for annotation in annotations)
    ranked_categories = [
        category_id
        for category_id, count in category_counts.most_common()
        if count >= args.min_boxes
        and (args.include_unknown or category_names[category_id] != "unknown_product")
    ]
    selected_category_ids = ranked_categories[: args.top_k]

    if not selected_category_ids:
        raise SystemExit("No classes matched the export settings.")

    class_names = [category_names[category_id] for category_id in selected_category_ids]
    class_index = {
        category_id: yolo_index for yolo_index, category_id in enumerate(selected_category_ids)
    }

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        category_id = annotation["category_id"]
        if category_id in class_index:
            annotations_by_image[annotation["image_id"]].append(annotation)

    selected_image_ids = sorted(annotations_by_image)
    if len(selected_image_ids) < 2:
        raise SystemExit("Need at least 2 images after filtering to create train/val splits.")

    rng = random.Random(args.seed)
    rng.shuffle(selected_image_ids)

    train_count = max(1, min(len(selected_image_ids) - 1, int(len(selected_image_ids) * args.train_ratio)))
    train_ids = set(selected_image_ids[:train_count])

    export_name = args.output_name or f"yolo_top{args.top_k}"
    export_root = OUTPUT_DIR / export_name
    for split in ("train", "val"):
        (export_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (export_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_id in selected_image_ids:
        image = image_lookup[image_id]
        source_path = IMAGES_DIR / image["file_name"]
        if not source_path.exists():
            raise SystemExit(f"Missing image file: {source_path}")

        split = "train" if image_id in train_ids else "val"
        target_image_path = export_root / "images" / split / image["file_name"]
        shutil.copy2(source_path, target_image_path)

        label_path = export_root / "labels" / split / f"{Path(image['file_name']).stem}.txt"
        with label_path.open("w", encoding="utf-8") as label_file:
            for annotation in annotations_by_image[image_id]:
                x_center, y_center, width, height = yolo_box(
                    annotation["bbox"], image["width"], image["height"]
                )
                label_file.write(
                    f"{class_index[annotation['category_id']]} "
                    f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )

    data_yaml = export_root / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as fh:
        fh.write(f"path: {export_root}\n")
        fh.write("train: images/train\n")
        fh.write("val: images/val\n")
        fh.write("names:\n")
        for index, name in enumerate(class_names):
            fh.write(f"  {index}: {name}\n")

    classes_txt = export_root / "classes.txt"
    with classes_txt.open("w", encoding="utf-8") as fh:
        for name in class_names:
            fh.write(f"{name}\n")

    summary_path = export_root / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"classes: {len(class_names)}\n")
        fh.write(f"images: {len(selected_image_ids)}\n")
        fh.write(f"train_images: {len(train_ids)}\n")
        fh.write(f"val_images: {len(selected_image_ids) - len(train_ids)}\n")
        fh.write("\nselected_classes:\n")
        for category_id in selected_category_ids:
            fh.write(f"- {category_names[category_id]} ({category_counts[category_id]} boxes)\n")

    print(f"Exported dataset to: {export_root}")
    print(f"Classes: {len(class_names)}")
    print(f"Images: {len(selected_image_ids)}")
    print(f"Train images: {len(train_ids)}")
    print(f"Val images: {len(selected_image_ids) - len(train_ids)}")


if __name__ == "__main__":
    main()
