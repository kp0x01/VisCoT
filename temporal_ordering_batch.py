#!/usr/bin/env python
"""Batch inference helper for the temporal ordering dataset."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

# Mirror the training setup by disabling flash attention kernels.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


class TemporalOrderingInference:
    """Thin wrapper that keeps the model/tokenizer in memory across calls."""

    def __init__(self, model_path: str) -> None:
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

    def infer(self, image_path: str, query: str) -> tuple[Optional[str], Optional[str]]:
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

            conv = conv_templates["vicuna_v1"].copy()
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{query}"
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
                    max_new_tokens=50,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            vocab_size = len(self.tokenizer)
            filtered_ids: List[torch.Tensor] = []
            for ids in output_ids:
                valid_ids = [token_id for token_id in ids.tolist() if 0 <= token_id < vocab_size]
                filtered_ids.append(torch.tensor(valid_ids, device=self.model.device))

            outputs = self.tokenizer.batch_decode(filtered_ids, skip_special_tokens=True)[0].strip()
            return outputs, None
        except Exception as exc:  # noqa: BLE001
            import traceback

            error_msg = f"{exc}\n{traceback.format_exc()}"
            return None, error_msg


def collect_images(data_dir: str, extensions: Optional[List[str]]) -> List[Path]:
    paths: List[Path] = []
    directory = Path(data_dir)
    exts = extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    for ext in exts:
        paths.extend(directory.glob(f"*{ext}"))
        paths.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(paths)


def parse_prediction(response: Optional[str]) -> str:
    # Mirror the original heuristic exactly; we intentionally let it fail if
    # the response is None to avoid changing behavior.
    if "ANSWER:" in response:
        first_word = response.split("ASSISTANT:")[1][:20]
        return "first" if "first" in first_word.lower() else "second"
    return "NA"


def process_dataset(
    model_path: str,
    data_dir: str,
    output_file: str,
    query: str,
    image_extensions: Optional[List[str]] = None,
) -> None:
    images = collect_images(data_dir, image_extensions)
    print(f"Found {len(images)} images in {data_dir}")
    if not images:
        print(f"ERROR: No images found in {data_dir}")
        return

    inferencer = TemporalOrderingInference(model_path)
    results: List[dict] = []

    for image_path in tqdm(images, desc="Processing images"):
        response, error = inferencer.infer(str(image_path), query)
        prediction = parse_prediction(response)
        results.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "query": query,
                "prediction": prediction,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if len(results) % 10 == 0:
            with open(output_file, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)
            print(f"\nSaved checkpoint at {len(results)} images")

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    successful = sum(1 for r in results if r["error"] is None)
    failed = len(results) - successful
    first_count = sum(1 for r in results if r["prediction"] == "first")
    second_count = sum(1 for r in results if r["prediction"] == "second")

    print(f"\n{'=' * 80}")
    print("Processing complete!")
    print(f"Total images: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Predictions - 'first': {first_count}, 'second': {second_count}")
    print(f"Results saved to: {output_file}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal Ordering Inference")
    parser.add_argument("--model-path", type=str, default="checkpoints/VisCoT-7b-336")
    parser.add_argument("--data-dir", type=str, default="/workspace/data/temporal_concat")
    parser.add_argument("--output", type=str, default="temporal_ordering_results.json")
    parser.add_argument(
        "--query",
        type=str,
        default=(
            "TASK: Determine temporal order of LEFT and RIGHT images. STEP 1: Describe key differences "
            "between the images (look for: burnt/cut objects, people positions, subjects entering "
            "and leaving the scene, sun/shadow positions, object states, effects of gravity etc.). "
            "STEP 2: Analyze which state came first chronologically. STEP 3: State your final answer. "
            "FORMAT: Write \"ANSWER: first\" if LEFT happened earlier, or \"ANSWER: second\" if RIGHT "
            "happened earlier. Then explain your reasoning in 1-2 sentences."
        ),
    )
    parser.add_argument("--ext", type=str, nargs="*", default=None, help="Optional list of allowed extensions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.data_dir).exists():
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        return
    process_dataset(args.model_path, args.data_dir, args.output, args.query, args.ext)


if __name__ == "__main__":
    main()
