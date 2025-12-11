import argparse
import json
from pathlib import Path
from typing import List, Tuple, Union

from PIL import Image


def load_items(path: Path) -> List[dict]:
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        items = data["data"]
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")
    return [item for item in items if item.get("task") == "visual_ordering"]


def parse_recipe_name(filename: str) -> str:
    """
    Extract recipe name from a filename like 'how-to-cook-fava-beans_3_0.jpg'.
    Returns a human-friendly string (hyphens/underscores replaced with spaces).
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        recipe_id = "_".join(parts[:-2])
    else:
        recipe_id = stem
    # Make it human readable
    return recipe_id.replace("-", " ").replace("_", " ").strip()


def concat_four_images(img_paths: List[Path]) -> Image.Image:
    """Horizontally concatenate four images after resizing to a common height."""
    images = [Image.open(p).convert("RGB") for p in img_paths]
    target_h = max(img.height for img in images)

    resized = []
    for img in images:
        if img.height != target_h:
            new_w = int(img.width * (target_h / img.height))
            img = img.resize((new_w, target_h), Image.LANCZOS)
        resized.append(img)

    total_w = sum(img.width for img in resized)
    canvas = Image.new("RGB", (total_w, target_h), (255, 255, 255))

    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge RecipeQA visual_ordering choices into strip images and build train/val JSON."
    )
    parser.add_argument("--input", type=Path, default=Path("filtered.json"), help="Filtered JSON path.")
    parser.add_argument("--image-folder", type=Path, default=Path("recipeqa/test"), help="Source images folder.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("organize_recipeqa"), help="Destination root folder."
    )
    parser.add_argument(
        "--train-json", type=Path, default=Path("organize_recipeqa_train.json"), help="Output train JSON path."
    )
    parser.add_argument(
        "--val-json", type=Path, default=Path("organize_recipeqa_val.json"), help="Output val JSON path."
    )
    parser.add_argument(
        "--train-size", type=int, default=50, help="Number of items for train split (default 50)."
    )
    parser.add_argument("--val-size", type=int, default=20, help="Number of items for val split (default 20).")
    parser.add_argument(
        "--max-ids",
        type=int,
        default=70,
        help="Max number of items to process (each yields 4 images). Use -1 for all.",
    )
    args = parser.parse_args()

    items = load_items(args.input)
    if args.max_ids and args.max_ids > 0:
        items = items[: args.max_ids]

    train_items = items[: args.train_size]
    val_items = items[args.train_size : args.train_size + args.val_size]

    # Ensure output dirs exist
    (args.output_dir / "train").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "val").mkdir(parents=True, exist_ok=True)

    def process_split(
        split_name: str, split_items: List[dict], base_idx: int, out_dir: Path
    ) -> Tuple[int, List[dict]]:
        saved = 0
        records: List[dict] = []
        for offset, item in enumerate(split_items):
            sample_idx = base_idx + offset
            choices: Union[list, None] = item.get("choice_list")
            if not isinstance(choices, list):
                print(f"[Skip] item {sample_idx} missing choice_list")
                continue
            answer_idx = int(item.get("answer", -1))

            for choice_idx, choice_imgs in enumerate(choices):
                if not isinstance(choice_imgs, list) or len(choice_imgs) != 4:
                    print(f"[Skip] item {sample_idx} choice {choice_idx} not 4 images")
                    continue

                paths = [args.image_folder / name for name in choice_imgs]
                if not all(p.exists() for p in paths):
                    missing = [p.name for p in paths if not p.exists()]
                    print(f"[Skip] item {sample_idx} choice {choice_idx} missing: {missing}")
                    continue

                merged = concat_four_images(paths)

                is_correct = choice_idx == answer_idx
                filename = f"{sample_idx}_{choice_idx}_{'true' if is_correct else 'false'}.jpg"
                merged_path = out_dir / filename
                merged.save(merged_path, format="JPEG", quality=95)
                saved += 1

                recipe_name = parse_recipe_name(choice_imgs[0])
                human_text = (
                    f"Identify if from left to right is the correct order of cooking this {recipe_name}. "
                    f"Answer only 'true' or 'false'."
                )
                records.append(
                    {
                        "id": f"recipeqa_{sample_idx}_{choice_idx}",
                        "image": filename,
                        "conversations": [
                            {"from": "human", "value": human_text},
                            {"from": "gpt", "value": "true" if is_correct else "false"},
                        ],
                    }
                )
        return saved, records

    train_saved, train_records = process_split("train", train_items, 1, args.output_dir / "train")
    val_saved, val_records = process_split(
        "val", val_items, 1 + len(train_items), args.output_dir / "val"
    )

    with args.train_json.open("w") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with args.val_json.open("w") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    print(f"Train: saved {train_saved} images -> {args.output_dir / 'train'} ; records: {len(train_records)}")
    print(f"Val:   saved {val_saved} images -> {args.output_dir / 'val'}   ; records: {len(val_records)}")
    print(f"Wrote JSON: {args.train_json}, {args.val_json}")


if __name__ == "__main__":
    main()
