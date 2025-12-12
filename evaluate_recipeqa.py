#!/usr/bin/env python
"""RecipeQA evaluation script for the LoRA adapter."""
from __future__ import annotations

import json

import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token

PROMPT = (
    "Look at these four video frames shown in sequence from left to right. Are they in the correct temporal order? "
    "Answer 'true' if they are in the correct chronological order, or 'false' if they are shuffled or out of order."
)

print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained("checkpoints/VisCoT-7b-336", use_fast=False)
model = LlavaLlamaForCausalLM.from_pretrained(
    "checkpoints/VisCoT-7b-336",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

CHECKPOINT_PATH = "recipeqa_lora_output/checkpoint-30"
model = PeftModel.from_pretrained(model, CHECKPOINT_PATH, is_trainable=False)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    tower = CLIPVisionTower("openai/clip-vit-large-patch14-336")
    tower.load_model()
    model.get_model().vision_tower = tower
    vision_tower = tower

vision_tower.to(device="cuda", dtype=torch.float16)
image_processor = vision_tower.image_processor
model.eval()

with open("recipeqa_val.json", "r", encoding="utf-8") as handle:
    test_data = json.load(handle)

print(f"Evaluating on {len(test_data)} test samples...")


def predict(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{PROMPT}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            max_new_tokens=10,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    return output


def main() -> None:
    ground_truths: list[str] = []
    predictions: list[str] = []
    results: list[dict] = []

    for sample in tqdm(test_data):
        image_path = sample["image"]
        gt = sample["conversations"][1]["value"].lower().strip()
        pred_raw = predict(image_path)
        print(f"Raw prediction: {pred_raw}")
        if "true" in pred_raw:
            pred = "true"
        elif "false" in pred_raw:
            pred = "false"
        else:
            pred = "false"

        ground_truths.append(gt)
        predictions.append(pred)
        is_correct = pred == gt
        results.append(
            {
                "image": image_path,
                "ground_truth": gt,
                "prediction": pred,
                "raw_prediction": pred_raw,
                "correct": is_correct,
            }
        )
        if len(results) <= 5:
            print(f"\nImage: {image_path}")
            print(f"GT: {gt}, Pred: {pred}, Correct: {is_correct}")

    accuracy = accuracy_score(ground_truths, predictions)
    y_true = [1 if gt == "true" else 0 for gt in ground_truths]
    y_pred = [1 if pred == "true" else 0 for pred in predictions]

    precision_macro = precision_score(y_true, y_pred, average="macro")
    recall_macro = recall_score(y_true, y_pred, average="macro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 70}")
    print("RecipeQA EVALUATION RESULTS")
    print("=" * 70)
    print(f"Test Accuracy: {accuracy:.2%} ({int(accuracy * len(test_data))}/{len(test_data)})")
    print("\nMacro Averages:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall:    {recall_macro:.4f}")
    print(f"  F1 Score:  {f1_macro:.4f}")

    true_total = sum(1 for gt in ground_truths if gt == "true")
    true_correct = sum(1 for gt, pred in zip(ground_truths, predictions) if gt == "true" and pred == "true")
    false_total = sum(1 for gt in ground_truths if gt == "false")
    false_correct = sum(1 for gt, pred in zip(ground_truths, predictions) if gt == "false" and pred == "false")

    print(f"\n{'=' * 70}")
    print("PER-CLASS METRICS")
    print("=" * 70)
    print("\n'true' class (correct order):")
    print(f"  Accuracy:  {true_correct}/{true_total} = {true_correct / true_total:.2%}")
    print(f"  Precision: {precision_per_class[1]:.4f}")
    print(f"  Recall:    {recall_per_class[1]:.4f}")
    print(f"  F1 Score:  {f1_per_class[1]:.4f}")
    print("\n'false' class (shuffled order):")
    print(f"  Accuracy:  {false_correct}/{false_total} = {false_correct / false_total:.2%}")
    print(f"  Precision: {precision_per_class[0]:.4f}")
    print(f"  Recall:    {recall_per_class[0]:.4f}")
    print(f"  F1 Score:  {f1_per_class[0]:.4f}")

    print(f"\n{'=' * 70}")
    print("CONFUSION MATRIX")
    print("=" * 70)
    print(f"\n                Predicted 'false'  Predicted 'true'")
    print(f"Actual 'false'        {cm[0, 0]:4d}              {cm[0, 1]:4d}")
    print(f"Actual 'true'         {cm[1, 0]:4d}              {cm[1, 1]:4d}")

    print(f"\n{'=' * 70}")
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=["false", "true"], digits=4))

    pd.DataFrame(results).to_csv("recipeqa_test_results.csv", index=False)
    print("\nResults saved to recipeqa_test_results.csv")


if __name__ == "__main__":
    main()
