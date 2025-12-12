#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from PIL import Image
import torch

# Disable flash attention kernels so we match the training environment.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

ModelBundle = Tuple[object, torch.nn.Module, object, int]


def load_model_bundle(model_path: Path) -> ModelBundle:
    """Load tokenizer/model/processor once and return the pieces."""
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=str(model_path),
        model_base=None,
        model_name="llava-v1.5-7b",
        load_8bit=False,
        load_4bit=False,
    )
    return tokenizer, model.to(dtype=torch.float16), image_processor, context_len


def build_prompt(query: str) -> str:
    conv = conv_templates["vicuna_v1"].copy()
    question = query if query.startswith(DEFAULT_IMAGE_TOKEN) else f"{DEFAULT_IMAGE_TOKEN}\n{query}"
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def decode_sequences(tokenizer, sequences: torch.Tensor) -> str:
    vocab_size = len(tokenizer)
    cleaned = []
    for seq in sequences:
        filtered = [tok_id for tok_id in seq.tolist() if 0 <= tok_id < vocab_size]
        cleaned.append(torch.tensor(filtered))
    return tokenizer.batch_decode(cleaned, skip_special_tokens=True)[0].strip()


def run_inference(model_path: Path, image_path: Path, query: str, temperature: float, max_tokens: int) -> str:
    tokenizer, model, image_processor, context_len = load_model_bundle(model_path)
    print(f"Loaded Visual-CoT checkpoint ({context_len=} tokens).")

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    prompt = build_prompt(query)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(model.device)
    sampling_temp = max(float(temperature), 1e-3)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=sampling_temp > 0,
            temperature=sampling_temp,
            max_new_tokens=max_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return decode_sequences(tokenizer, output_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual-CoT single-image inference helper.")
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/VisCoT-7b-336"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(args.image)
    result = run_inference(args.model_path, args.image, args.query, args.temperature, args.max_tokens)
    separator = "=" * 80
    print(separator)
    print("RESPONSE")
    print(separator)
    print(result)
    print(separator)


if __name__ == "__main__":
    main()
