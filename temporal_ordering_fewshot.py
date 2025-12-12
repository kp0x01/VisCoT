#!/usr/bin/env python
"""Few-shot inference harness that can stitch example panels above the query."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

# Keep runtime consistent with the multi-GPU training environment.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


class FewShotTemporalInference:
    """Wraps model loading and optional few-shot exemplar rendering."""

    def __init__(self, model_path: str, example_dir: Optional[str] = None) -> None:
        print(f"Loading model from {model_path}...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name="llava-v1.5-7b",
            load_8bit=False,
            load_4bit=False,
        )
        self.tokenizer = tokenizer
        self.model = model.to(dtype=torch.float16)
        self.image_processor = image_processor
        self.context_len = context_len
        print(f"Model loaded! Vocab size: {len(self.tokenizer)}")

        self.example_images: List[str] = []
        self.example_labels: List[str] = []
        if example_dir:
            self.load_examples(example_dir)

    def load_examples(self, example_dir: str) -> None:
        example_path = Path(example_dir)
        examples_file = example_path / "examples.json"
        if not examples_file.exists():
            print(f"Warning: {examples_file} not found. No examples loaded.")
            return

        with open(examples_file, "r", encoding="utf-8") as handle:
            examples_data = json.load(handle)
        for record in examples_data:
            img_path = example_path / record["image"]
            if img_path.exists():
                self.example_images.append(str(img_path))
                self.example_labels.append(record["label"])
        print(f"Loaded {len(self.example_images)} example images")

    def create_few_shot_image(self, query_image_path: str, num_shots: int = 2) -> Image.Image:
        query_img = Image.open(query_image_path).convert("RGB")
        example_imgs = [Image.open(p).convert("RGB") for p in self.example_images[:num_shots]]
        if not example_imgs:
            return query_img

        max_width = max(img.width for img in example_imgs + [query_img])
        resized: List[Image.Image] = []
        for img in example_imgs + [query_img]:
            aspect = img.height / img.width
            resized.append(img.resize((max_width, int(max_width * aspect)), Image.LANCZOS))

        label_height = 50
        total_height = sum(img.height + label_height for img in resized)
        canvas = Image.new("RGB", (max_width, total_height), color="white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 36)
        except OSError:
            font = ImageFont.load_default()

        y_offset = 0
        for idx, (img, label) in enumerate(zip(resized[:-1], self.example_labels[:num_shots])):
            canvas.paste(img, (0, y_offset))
            y_offset += img.height
            draw.text((10, y_offset + 10), f"Example {idx + 1}: Answer = {label}", fill="black", font=font)
            y_offset += label_height

        canvas.paste(resized[-1], (0, y_offset))
        y_offset += resized[-1].height
        draw.text((10, y_offset + 10), "Query: Answer = ?", fill="red", font=font)
        return canvas

    def infer(self, image_path: str, query: str, use_few_shot: bool = False, num_shots: int = 2):
        try:
            if use_few_shot and self.example_images:
                image = self.create_few_shot_image(image_path, num_shots)
                enhanced_query = (
                    f"Look at the images above. I've shown you {num_shots} example(s) with their correct answers.\n\n"
                    f"Now answer this query image (marked with \"?\") using the same reasoning:\n{query}"
                )
            else:
                image = Image.open(image_path).convert("RGB")
                enhanced_query = query

            image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

            conv = conv_templates["vicuna_v1"].copy()
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{enhanced_query}"
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt_text,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=100,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            vocab_size = len(self.tokenizer)
            filtered_ids: List[torch.Tensor] = []
            for ids in output_ids:
                valid_ids = [tid for tid in ids.tolist() if 0 <= tid < vocab_size]
                filtered_ids.append(torch.tensor(valid_ids, device=self.model.device))

            outputs = self.tokenizer.batch_decode(filtered_ids, skip_special_tokens=True)[0].strip()
            response_lower = outputs.lower()
            # Preserve original behavior by returning the raw string even though we
            # compute lower-cased helper variables.
            result = outputs
            return result, None
        except Exception as exc:  # noqa: BLE001
            import traceback

            return None, f"{exc}\n{traceback.format_exc()}"


def process_dataset(
    model_path: str,
    data_dir: str,
    output_file: str,
    query: str,
    example_dir: Optional[str] = None,
    use_few_shot: bool = False,
    num_shots: int = 2,
) -> None:
    data_path = Path(data_dir)
    image_files = sorted(list(data_path.glob("*.jpg")) + list(data_path.glob("*.png")) + list(data_path.glob("*.jpeg")))
    print(f"Found {len(image_files)} images in {data_dir}")
    if not image_files:
        print("ERROR: No images found")
        return

    inferencer = FewShotTemporalInference(model_path, example_dir)
    results: List[dict] = []

    for image_path in tqdm(image_files, desc="Processing"):
        response, error = inferencer.infer(str(image_path), query, use_few_shot, num_shots)
        results.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "query": query,
                "use_few_shot": use_few_shot,
                "num_shots": num_shots if use_few_shot else 0,
                "prediction": response,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if len(results) % 10 == 0:
            with open(output_file, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    successful = sum(1 for r in results if r["error"] is None)
    first_count = sum(1 for r in results if r["prediction"] == "first")
    second_count = sum(1 for r in results if r["prediction"] == "second")

    print(f"\n{'=' * 80}")
    print(f"Complete! Total: {len(results)} | Success: {successful}")
    print(f"Predictions - first: {first_count} | second: {second_count}")
    print(f"Results: {output_file}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-Shot Temporal Ordering with Images")
    parser.add_argument("--model-path", type=str, default="checkpoints/VisCoT-7b-336")
    parser.add_argument("--data-dir", type=str, default="/workspace/data/temporal_concat")
    parser.add_argument("--example-dir", type=str, default=None, help="Directory with example images and examples.json")
    parser.add_argument("--output", type=str, default="few_shot_results.json")
    parser.add_argument(
        "--query",
        type=str,
        default=(
            "TASK: Temporal ordering of two images (LEFT and RIGHT). STEP 1: Describe what changed between images. "
            "STEP 2: Determine which state came first chronologically. STEP 3: If LEFT is earlier state, answer \"first\". "
            "If RIGHT is earlier state, answer \"second\". FORMAT: ANSWER: [first or second] REASONING: [one sentence why]"
        ),
    )
    parser.add_argument("--few-shot", action="store_true", help="Use few-shot mode with example images")
    parser.add_argument("--num-shots", type=int, default=2, help="Number of example images to include")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return
    process_dataset(
        args.model_path,
        args.data_dir,
        args.output,
        args.query,
        args.example_dir,
        args.few_shot,
        args.num_shots,
    )


if __name__ == "__main__":
    main()
