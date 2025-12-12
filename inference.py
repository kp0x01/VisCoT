#!/usr/bin/env python
"""Tiny helper to run Visual-CoT inference on a single image."""
from __future__ import annotations
import argparse
from pathlib import Path

from PIL import Image
import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

# Disable flash attention kernels to match the training setup.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

DEFAULT_MODEL = Path("checkpoints/VisCoT-7b-336")


def run_inference(
    model_path: Path,
    image_path: Path,
    query: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """Load the checkpoint, run one forward pass, and return the decoded response."""
    print(f"Loading model from {model_path}...")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=str(model_path),
        model_base=None,
        model_name="llava-v1.5-7b",
        load_8bit=False,
        load_4bit=False,
    )
    model = model.to(dtype=torch.float16)

    print(f"Model loaded! Context length: {context_len}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Processing image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    conv = conv_templates["vicuna_v1"].copy()
    user_turn = f"{DEFAULT_IMAGE_TOKEN}\n{query}"
    conv.append_message(conv.roles[0], user_turn)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    print(f"\nQuery: {query}")
    print("Generating response...\n")

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=max_tokens,
            use_cache=True,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    vocab_size = len(tokenizer)
    valid_output_ids = []
    for ids in output_ids:
        valid_ids = [tok for tok in ids.tolist() if 0 <= tok < vocab_size]
        valid_output_ids.append(torch.tensor(valid_ids))

    try:
        outputs = tokenizer.batch_decode(valid_output_ids, skip_special_tokens=True)[0].strip()
    except Exception as exc:  # noqa: BLE001
        print(f"Decoding error: {exc}")
        outputs = tokenizer.batch_decode(valid_output_ids, skip_special_tokens=False)[0].strip()

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual-CoT Inference")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL, help="Checkpoint directory")
    parser.add_argument("--image", type=Path, required=True, help="Image to evaluate")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum generation length")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(args.image)

    result = run_inference(args.model_path, args.image, args.query, args.temperature, args.max_tokens)

    print("=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(result)
    print("=" * 80)


if __name__ == "__main__":
    main()
