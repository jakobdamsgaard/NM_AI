#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = ROOT / "annotations.json"


def main() -> None:
    with ANNOTATIONS_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    category_names = {category["id"]: category["name"] for category in categories}
    counts = Counter(annotation["category_id"] for annotation in annotations)

    print("Dataset summary")
    print(f"- Images: {len(images)}")
    print(f"- Annotations: {len(annotations)}")
    print(f"- Classes: {len(categories)}")
    print()

    print("Top 20 classes by annotation count")
    for rank, (category_id, count) in enumerate(counts.most_common(20), start=1):
        print(f"{rank:>2}. {count:>4}  {category_names[category_id]}")

    print()
    image_ids = {image["id"] for image in images}
    annotated_image_ids = {annotation["image_id"] for annotation in annotations}
    print(f"Images with annotations: {len(image_ids & annotated_image_ids)}")
    print(f"Images without annotations: {len(image_ids - annotated_image_ids)}")

    print()
    print("Recommendation")
    print("- Start with the top 10-20 classes.")
    print("- Exclude 'unknown_product' in your first training run.")
    print("- Export a smaller YOLO dataset before training.")


if __name__ == "__main__":
    main()
