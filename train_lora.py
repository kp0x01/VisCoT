#!/usr/bin/env python
"""Standalone script to fine-tune LLaVA with LoRA adapters on the temporal dataset."""
from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "llava-hf/llava-1.5-7b-hf"
OUTPUT_DIR = "./temporal_lora_output"
TRAIN_JSON = "temporal_train.json"
VAL_JSON = "temporal_val.json"
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 20
MAX_LENGTH = 512
WARMUP_RATIO = 0.1
USE_QLORA = True
PROMPT = (
    "TASK: Determine temporal order of LEFT and RIGHT images. \n"
    "    STEP 1: Describe key differences between the images (look for: burnt/cut objects, people positions, sun/shadow positions, object states, etc.)\n"
    "    STEP 2: Analyze which state came first chronologically.\n"
    "    STEP 3: State your final answer. FORMAT: Write \"ANSWER: first\" if LEFT happened earlier, or \"ANSWER: second\" if RIGHT happened earlier. Then explain your reasoning in 1-2 sentences."
)


class TemporalQADataset(torch.utils.data.Dataset):
    def __init__(self, json_path: str, processor, max_length: int = 512) -> None:
        with open(json_path, "r", encoding="utf-8") as handle:
            self.data = json.load(handle)
        self.processor = processor
        self.max_length = max_length
        print(f"Loaded {len(self.data)} samples from {json_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        try:
            image = Image.open(sample["image"]).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"Error loading image {sample['image']}: {exc}")
            image = Image.new("RGB", (224, 224), color="black")

        conversations = sample["conversations"]
        question = conversations[0]["value"]
        answer = conversations[1]["value"]
        text = f"USER: {question} ASSISTANT: {answer}"

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        labels = inputs["input_ids"].clone()
        assistant_str = "ASSISTANT:"
        assistant_tokens = self.processor.tokenizer.encode(assistant_str, add_special_tokens=False)
        input_ids_list = inputs["input_ids"].tolist()

        try:
            for i in range(len(input_ids_list) - len(assistant_tokens) + 1):
                if input_ids_list[i : i + len(assistant_tokens)] == assistant_tokens:
                    labels[: i + len(assistant_tokens)] = -100
                    break
        except Exception:  # preserve original fallback behavior
            labels[: len(labels) // 2] = -100

        inputs["labels"] = labels
        return inputs


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated: Dict[str, torch.Tensor] = {}
    for key in keys:
        if key in {"input_ids", "attention_mask", "labels", "pixel_values"}:
            collated[key] = torch.stack([item[key] for item in batch])
    return collated


def load_model_with_lora():
    print(f"Loading base model: {BASE_MODEL}")
    if USE_QLORA:
        print("Using 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("Using FP16 (no quantization)")
        model = LlavaForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")

    print(f"Adding LoRA adapters (r={LORA_R}, alpha={LORA_ALPHA})")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)\n"
    )
    return model


def train() -> None:
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.padding_side = "right"

    model = load_model_with_lora()

    print("\nLoading datasets...")
    train_dataset = TemporalQADataset(TRAIN_JSON, processor, MAX_LENGTH)
    val_dataset = TemporalQADataset(VAL_JSON, processor, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    trainer.train()

    final_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    print(f"\nSaving final model to {final_checkpoint_dir}")
    trainer.save_model(final_checkpoint_dir)
    processor.save_pretrained(final_checkpoint_dir)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nModel saved to: {final_checkpoint_dir}")
    print(f"TensorBoard logs: {OUTPUT_DIR}")
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
