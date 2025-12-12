#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from PIL import Image
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
class SampleResult:
    image_path: Path
    prediction: str
    raw_output: Optional[str]
    error: Optional[str]


class TemporalOrderingInference:
    def __init__(self, model_path: Path, lora_path: Optional[Path], merge_lora: bool) -> None:
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

    def _decode(self, output_ids: torch.Tensor) -> str:
        vocab_size = len(self.tokenizer)
        cleaned = []
        for seq in output_ids:
            filtered = [tok for tok in seq.tolist() if 0 <= tok < vocab_size]
            cleaned.append(torch.tensor(filtered, device=self.model.device))
        return self.tokenizer.batch_decode(cleaned, skip_special_tokens=True)[0].strip()

    def infer(self, image_path: Path, query: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = process_images([image], self.image_processor, self.model.config)
            tensor = tensor.to(self.model.device, dtype=torch.float16)

            prompt = self._prepare_prompt(query)
            input_ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=tensor,
                    do_sample=False,
                    max_new_tokens=50,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            return self._decode(output_ids), None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)


def collect_images(data_dir: Path, extensions: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for ext in extensions:
        files.extend(data_dir.glob(f"*{ext}"))
        files.extend(data_dir.glob(f"*{ext.upper()}"))
    return sorted(files)


def extract_binary_label(text: Optional[str]) -> str:
    if not text:
        return "NA"
    lowered = text.lower()
    if "assistant:" in lowered:
        lowered = lowered.split("assistant:", 1)[1].strip()
    first_word = lowered.split()[0] if lowered else ""
    return first_word if first_word in {"first", "second"} else "NA"


def write_results(path: Path, results: Sequence[SampleResult]) -> None:
    payload = [
        {
            "image_name": res.image_path.name,
            "image_path": str(res.image_path),
            "prediction": res.prediction,
            "raw_output": res.raw_output,
            "error": res.error,
            "timestamp": datetime.now().isoformat(),
        }
        for res in results
    ]
    path.write_text(json.dumps(payload, indent=2))


def maybe_checkpoint(output_path: Path, results: List[SampleResult], interval: int) -> None:
    if interval <= 0 or len(results) % interval:
        return
    write_results(output_path, results)
    print(f"[checkpoint] saved {len(results)} samples to {output_path}")


def process_dataset(
    model_path: Path,
    data_dir: Path,
    output_path: Path,
    query: str,
    lora_path: Optional[Path],
    merge_lora: bool,
    image_extensions: Sequence[str],
) -> None:
    images = collect_images(data_dir, image_extensions)
    if not images:
        raise FileNotFoundError(f"No images found under {data_dir}")

    runner = TemporalOrderingInference(model_path, lora_path, merge_lora)
    normalized_query = query if query.startswith(DEFAULT_IMAGE_TOKEN) else f"{DEFAULT_IMAGE_TOKEN}\n{query}"

    results: List[SampleResult] = []
    for image_path in tqdm(images, desc="Temporal ordering"):
        response, error = runner.infer(image_path, normalized_query)
        results.append(SampleResult(image_path=image_path, prediction=extract_binary_label(response), raw_output=response, error=error))
        maybe_checkpoint(output_path, results, interval=10)

    write_results(output_path, results)
    successes = sum(1 for res in results if res.error is None)
    print(f"Processed {len(results)} samples | success={successes} | output={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for the temporal ordering benchmark.")
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/viscot-temporal-prefix"))
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/data/temporal_concat"))
    parser.add_argument("--output", type=Path, default=Path("temporal_ordering_results.json"))
    parser.add_argument(
        "--query",
        type=str,
        default='<image>\nTASK: Determine temporal order of LEFT and RIGHT images. FORMAT: Reply with exactly one word: "first" or "second". Do not add reasoning.',
    )
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--ext", type=str, nargs="*", default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_path=args.output,
        query=args.query,
        lora_path=args.lora_path,
        merge_lora=args.merge_lora,
        image_extensions=args.ext,
    )


if __name__ == "__main__":
    main()
