#!/usr/bin/env python3
"""Utility to sanitize the temporal dataset for prefix tuning.

- Forces every assistant answer to be exactly "first" or "second".
- Optionally appends a format hint to the human prompt so the model learns
  to respond with a single word during training and evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

FORMAT_SUFFIX = '\nFORMAT: Reply with exactly one word: "first" or "second". Do not add reasoning.'
IMAGE_PREFIX = "<image>\n"


def normalize_answer(text: str) -> str:
    lowered = text.strip().lower()
    if 'first' in lowered and 'second' in lowered:
        # Ambiguous label: raise so the user can inspect.
        raise ValueError(f"Label contains both 'first' and 'second': {text}")
    if 'first' in lowered:
        return 'first'
    if 'second' in lowered:
        return 'second'
    raise ValueError(f"Label must mention 'first' or 'second': {text}")


def maybe_append_format(prompt: str, force_format_hint: bool) -> str:
    if not force_format_hint:
        return prompt
    if FORMAT_SUFFIX.strip() in prompt:
        return prompt
    if prompt.endswith('\n'):
        return prompt + FORMAT_SUFFIX
    return prompt + '\n' + FORMAT_SUFFIX


def ensure_image_prefix(prompt: str) -> str:
    """Guarantee every prompt begins with the <image> token marker once."""
    stripped = prompt.lstrip()
    if stripped.startswith(IMAGE_PREFIX) or prompt.startswith(IMAGE_PREFIX):
        return prompt
    # Preserve original leading whitespace by inserting at the beginning.
    return IMAGE_PREFIX + prompt if not prompt.startswith(IMAGE_PREFIX) else prompt


def process_file(path: Path, dry_run: bool, force_format_hint: bool) -> None:
    data: List[Dict[str, Any]] = json.loads(path.read_text())
    changed = False
    for sample in data:
        conversation = sample.get('conversations') or []
        if len(conversation) < 2:
            raise ValueError(f"Sample missing conversation pair: {sample}")
        question = conversation[0]['value']
        answer = conversation[1]['value']
        normalized = normalize_answer(answer)
        if normalized != answer:
            conversation[1]['value'] = normalized
            changed = True
        updated_question = ensure_image_prefix(question)
        updated_question = maybe_append_format(updated_question, force_format_hint)
        if updated_question != question:
            conversation[0]['value'] = updated_question
            changed = True
    if changed and not dry_run:
        path.write_text(json.dumps(data, indent=2))
    print(f"Processed {path} | samples={len(data)} | changed={'yes' if changed else 'no'}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Normalize temporal dataset answers.')
    parser.add_argument('dataset_dir', type=Path, nargs='?', default=Path('temporal'))
    parser.add_argument('--dry-run', action='store_true', help='Report changes without writing files')
    parser.add_argument('--format-hint', action='store_true', help='Append one-word format hint to each question')
    args = parser.parse_args()

    for split in ['train', 'val', 'test']:
        json_path = args.dataset_dir / f'temporal_{split}.json'
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping")
            continue
        process_file(json_path, args.dry_run, args.format_hint)

if __name__ == '__main__':
    main()
