#!/usr/bin/env python
"""Helper routines for converting RecipeQA image dumps to VisCoT format."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

RECIPEQA_DIR = Path("/path/to/recipeqa_final")  # Update as needed
TRAIN_DIR = RECIPEQA_DIR / "train"
TEST_DIR = RECIPEQA_DIR / "test"
PROMPT = (
    "Look at these four video frames shown in sequence from left to right. Are they in "
    "the correct temporal order? Answer 'true' if they are in the correct "
    "chronological order, or 'false' if they are shuffled or out of order."
)


def parse_filename(filename: str) -> Tuple[str | None, str | None, str | None]:
    """Return (video_id, permutation_id, label) for a file like '5_1_true.jpg'."""
    stem = filename.replace(".jpg", "")
    parts = stem.split("_")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None, None, None


def create_recipeqa_dataset(image_dir: Path, output_json: Path) -> List[dict]:
    """Walk `image_dir`, convert each JPG into a Visual-CoT conversation, and save."""
    data: List[dict] = []
    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
    print(f"Found {len(image_files)} images in {image_dir}")

    for image_file in image_files:
        video_id, perm_id, label = parse_filename(image_file)
        if video_id is None:
            print(f"Warning: Could not parse {image_file}")
            continue
        if label not in {"true", "false"}:
            print(f"Warning: Invalid label '{label}' in {image_file}")
            continue

        full_image_path = image_dir / image_file
        data.append(
            {
                "id": f"recipeqa_{video_id}_{perm_id}",
                "image": str(full_image_path),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{PROMPT}"},
                    {"from": "gpt", "value": label},
                ],
            }
        )

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    print(f"Created {len(data)} samples")
    print(f"Saved to: {output_json}")
    true_count = sum(1 for entry in data if entry["conversations"][1]["value"] == "true")
    false_count = len(data) - true_count
    print("\nLabel distribution:")
    print(f"  true:  {true_count} ({100 * true_count / len(data):.1f}%)")
    print(f"  false: {false_count} ({100 * false_count / len(data):.1f}%)")
    return data


def split_recipeqa_by_video(
    image_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split filenames by video id so different permutations don't leak between splits."""
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    video_groups: dict[str, List[str]] = {}
    for image_file in image_files:
        video_id, _, _ = parse_filename(image_file)
        if video_id is None:
            continue
        video_groups.setdefault(video_id, []).append(image_file)

    print(f"\nFound {len(video_groups)} unique videos")
    print(f"Average {len(image_files) / len(video_groups):.1f} permutations per video")

    video_ids = list(video_groups.keys())
    train_ids, temp_ids = train_test_split(
        video_ids,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state,
    )

    def expand(ids: List[str]) -> List[str]:
        files: List[str] = []
        for vid in ids:
            files.extend(video_groups[vid])
        return files

    train_images, val_images, test_images = expand(train_ids), expand(val_ids), expand(test_ids)

    print("\nSplit by video ID:")
    print(f"  Train: {len(train_ids)} videos, {len(train_images)} images")
    print(f"  Val:   {len(val_ids)} videos, {len(val_images)} images")
    print(f"  Test:  {len(test_ids)} videos, {len(test_images)} images")
    return train_images, val_images, test_images


def create_split_json(image_list: List[str], image_dir: Path, output_json: Path) -> None:
    """Write a list of filenames into a VisCoT-style JSON file."""
    payload = []
    for image_file in image_list:
        video_id, perm_id, label = parse_filename(image_file)
        if video_id is None or label is None:
            continue
        full_path = image_dir / image_file
        payload.append(
            {
                "id": f"recipeqa_{video_id}_{perm_id}",
                "image": str(full_path),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{PROMPT}"},
                    {"from": "gpt", "value": label},
                ],
            }
        )

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Created {len(payload)} samples in {output_json}")


def main() -> None:
    image_dir = Path("/ocean/projects/cis250266p/kanand/data/recipeqa_final/train")

    if TRAIN_DIR.exists() and TEST_DIR.exists():
        print("Using existing train/test directories...")
        create_recipeqa_dataset(TRAIN_DIR, Path("recipeqa_train.json"))
        create_recipeqa_dataset(TEST_DIR, Path("recipeqa_test.json"))
    else:
        print("Creating custom split...")
        train_images, val_images, test_images = split_recipeqa_by_video(image_dir)
        os.makedirs("recipeqa_splits/train", exist_ok=True)
        os.makedirs("recipeqa_splits/val", exist_ok=True)
        os.makedirs("recipeqa_splits/test", exist_ok=True)
        create_split_json(train_images, image_dir, Path("recipeqa_train.json"))
        create_split_json(val_images, image_dir, Path("recipeqa_val.json"))
        create_split_json(test_images, image_dir, Path("recipeqa_test.json"))

    print("\n✓ RecipeQA data preparation complete!")


if __name__ == "__main__":
    main()
