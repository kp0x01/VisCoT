#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


@dataclass
class ExampleItem:
    path: Path
    label: str


class ExampleLibrary:
    def __init__(self, directory: Optional[Path]) -> None:
        self.directory = directory
        self.examples: List[ExampleItem] = []
        if directory:
            self._load_examples(directory)

    def _load_examples(self, directory: Path) -> None:
        manifest = directory / "examples.json"
        if not manifest.exists():
            print(f"Warning: {manifest} missing; continuing without few-shot examples")
            return
        data = json.loads(manifest.read_text())
        for entry in data:
            image_path = directory / entry["image"]
            if image_path.exists():
                self.examples.append(ExampleItem(path=image_path, label=entry["label"]))
        print(f"Loaded {len(self.examples)} reference example(s) from {directory}")

    def take(self, count: int) -> List[ExampleItem]:
        return self.examples[:count]


class FewShotTemporalInference:
    def __init__(
        self,
        model_path: Path,
        examples: ExampleLibrary,
        lora_path: Optional[Path],
        merge_lora: bool,
    ) -> None:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=str(model_path),
            model_base=None,
            model_name="llava-v1.5-7b",
            load_8bit=False,
            load_4bit=False,
        )
        self.tokenizer = tokenizer
        self.model = model.to(dtype=torch.float16)
        self.image_processor = image_processor
        self.examples = examples
        if lora_path:
            if PeftModel is None:
                raise ImportError("Install `peft` to load LoRA checkpoints.")
            adapter = PeftModel.from_pretrained(self.model, str(lora_path))
            if merge_lora:
                adapter = adapter.merge_and_unload()
            self.model = adapter.to(dtype=torch.float16)

    def _prepare_prompt(self, query: str) -> str:
        conv = conv_templates["vicuna_v1"].copy()
        question = query if query.startswith(DEFAULT_IMAGE_TOKEN) else f"{DEFAULT_IMAGE_TOKEN}\n{query}"
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    @staticmethod
    def _load_font(size: int) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def _compose_canvas(self, query_image: Path, num_shots: int) -> Image.Image:
        base = Image.open(query_image).convert("RGB")
        shots = self.examples.take(num_shots)
        if not shots:
            return base

        images = [Image.open(item.path).convert("RGB") for item in shots] + [base]
        target_width = max(img.width for img in images)
        resized: List[Image.Image] = []
        for img in images:
            aspect = img.height / img.width
            resized.append(img.resize((target_width, int(target_width * aspect)), Image.LANCZOS))

        label_height = 48
        total_height = sum(img.height + label_height for img in resized)
        canvas = Image.new("RGB", (target_width, total_height), color="white")
        draw = ImageDraw.Draw(canvas)
        font = self._load_font(32)

        y = 0
        for idx, (img, example) in enumerate(zip(resized[:-1], shots)):
            canvas.paste(img, (0, y))
            y += img.height
            draw.text((10, y + 12), f"Example {idx + 1}: {example.label}", fill="black", font=font)
            y += label_height

        # Query image block
        canvas.paste(resized[-1], (0, y))
        y += resized[-1].height
        draw.text((10, y + 12), "Query image", fill="red", font=font)
        return canvas

    def _decode(self, output_ids: torch.Tensor) -> str:
        vocab_size = len(self.tokenizer)
        cleaned = []
        for seq in output_ids:
            filtered = [tok for tok in seq.tolist() if 0 <= tok < vocab_size]
            cleaned.append(torch.tensor(filtered, device=self.model.device))
        return self.tokenizer.batch_decode(cleaned, skip_special_tokens=True)[0].strip()

    def infer(self, image_path: Path, query: str, use_few_shot: bool, num_shots: int) -> Tuple[Optional[str], Optional[str]]:
        try:
            if use_few_shot and self.examples.examples:
                composed = self._compose_canvas(image_path, num_shots)
                prefix = f"Consider the annotated examples above before answering the query image."
                prompt = f"{prefix}\n{query}"
            else:
                composed = Image.open(image_path).convert("RGB")
                prompt = query

            tensor = process_images([composed], self.image_processor, self.model.config)
            tensor = tensor.to(self.model.device, dtype=torch.float16)

            prepared_prompt = self._prepare_prompt(prompt)
            input_ids = tokenizer_image_token(
                prepared_prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=tensor,
                    do_sample=False,
                    max_new_tokens=100,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            return self._decode(output_ids), None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)


def list_images(root: Path, extensions: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for ext in extensions:
        files.extend(root.glob(f"*{ext}"))
        files.extend(root.glob(f"*{ext.upper()}"))
    return sorted(files)


def process_dataset(
    model_path: Path,
    data_dir: Path,
    example_dir: Optional[Path],
    output_path: Path,
    query: str,
    use_few_shot: bool,
    num_shots: int,
    lora_path: Optional[Path],
    merge_lora: bool,
    extensions: Sequence[str],
) -> None:
    images = list_images(data_dir, extensions)
    if not images:
        raise FileNotFoundError(f"No images found under {data_dir}")

    examples = ExampleLibrary(example_dir)
    runner = FewShotTemporalInference(model_path, examples, lora_path, merge_lora)

    normalized_query = query if query.startswith(DEFAULT_IMAGE_TOKEN) else f"{DEFAULT_IMAGE_TOKEN}\n{query}"
    results = []
    for image_path in tqdm(images, desc="Few-shot temporal ordering"):
        response, error = runner.infer(image_path, normalized_query, use_few_shot, num_shots)
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
            output_path.write_text(json.dumps(results, indent=2))

    output_path.write_text(json.dumps(results, indent=2))
    successes = sum(1 for row in results if row["error"] is None)
    print(f"Processed {len(results)} samples | success={successes} | output={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot inference helper for the temporal ordering benchmark.")
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/VisCoT-7b-336"))
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/data/temporal_concat"))
    parser.add_argument("--example-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("few_shot_results.json"))
    parser.add_argument(
        "--query",
        type=str,
        default='TASK: Temporal ordering of two images (LEFT and RIGHT). STEP 1: Describe what changed between images. STEP 2: Determine which state came first chronologically. STEP 3: If LEFT is earlier state, answer "first". If RIGHT is earlier state, answer "second". FORMAT: ANSWER: [first or second] REASONING: [one sentence why]',
    )
    parser.add_argument("--few-shot", action="store_true")
    parser.add_argument("--num-shots", type=int, default=2)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--ext", type=str, nargs="*", default=[".jpg", ".jpeg", ".png"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(
        model_path=args.model_path,
        data_dir=args.data_dir,
        example_dir=args.example_dir,
        output_path=args.output,
        query=args.query,
        use_few_shot=args.few_shot,
        num_shots=args.num_shots,
        lora_path=args.lora_path,
        merge_lora=args.merge_lora,
        extensions=args.ext,
    )


if __name__ == "__main__":
    main()
