# evaluate_recipeqa.py
import torch
from transformers import AutoTokenizer
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
from peft import PeftModel
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, 
    precision_score, recall_score, accuracy_score
)
import pandas as pd

PROMPT = "Look at these four video frames shown in sequence from left to right. Are they in the correct temporal order? Answer 'true' if they are in the correct chronological order, or 'false' if they are shuffled or out of order."

print("Loading model...")
from transformers import BitsAndBytesConfig

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
    torch_dtype=torch.float16
)

# Load LoRA weights
CHECKPOINT_PATH = "recipeqa_lora_output/checkpoint-30"  # Update with actual checkpoint
model = PeftModel.from_pretrained(model, CHECKPOINT_PATH, is_trainable=False)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    vision_tower = CLIPVisionTower("openai/clip-vit-large-patch14-336")
    vision_tower.load_model()
    model.get_model().vision_tower = vision_tower

vision_tower.to(device='cuda', dtype=torch.float16)
image_processor = vision_tower.image_processor
model.eval()

# Load test data
with open("recipeqa_val.json", 'r') as f:
    test_data = json.load(f)

print(f"Evaluating on {len(test_data)} test samples...")

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + PROMPT)
    conv.append_message(conv.roles[1], None)
    
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            max_new_tokens=10,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    return output

# Evaluate
ground_truths = []
predictions = []
results = []

for sample in tqdm(test_data):
    image_path = sample['image']
    gt = sample['conversations'][1]['value'].lower().strip()
    pred_raw = predict(image_path)
    print(f"Raw prediction: {pred_raw}")
    # Extract answer
    if 'true' in pred_raw:
        pred = 'true'
    elif 'false' in pred_raw:
        pred = 'false'
    else:
        pred = 'false'  # Default
    
    ground_truths.append(gt)
    predictions.append(pred)
    
    is_correct = (pred == gt)
    
    results.append({
        'image': image_path,
        'ground_truth': gt,
        'prediction': pred,
        'raw_prediction': pred_raw,
        'correct': is_correct
    })
    
    if len(results) <= 5:
        print(f"\nImage: {image_path}")
        print(f"GT: {gt}, Pred: {pred}, Correct: {is_correct}")

# Calculate metrics
accuracy = accuracy_score(ground_truths, predictions)

# Binary: 'true' as positive (1), 'false' as negative (0)
y_true = [1 if gt == 'true' else 0 for gt in ground_truths]
y_pred = [1 if pred == 'true' else 0 for pred in predictions]

precision_macro = precision_score(y_true, y_pred, average='macro')
recall_macro = recall_score(y_true, y_pred, average='macro')
f1_macro = f1_score(y_true, y_pred, average='macro')

precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)
f1_per_class = f1_score(y_true, y_pred, average=None)

cm = confusion_matrix(y_true, y_pred)

# Print results
print(f"\n{'='*70}")
print(f"RecipeQA EVALUATION RESULTS")
print(f"{'='*70}")
print(f"Test Accuracy: {accuracy:.2%} ({int(accuracy*len(test_data))}/{len(test_data)})")
print(f"\nMacro Averages:")
print(f"  Precision: {precision_macro:.4f}")
print(f"  Recall:    {recall_macro:.4f}")
print(f"  F1 Score:  {f1_macro:.4f}")

print(f"\n{'='*70}")
print(f"PER-CLASS METRICS")
print(f"{'='*70}")

# 'true' class
true_total = sum(1 for gt in ground_truths if gt == 'true')
true_correct = sum(1 for gt, pred in zip(ground_truths, predictions) if gt == 'true' and pred == 'true')
print(f"\n'true' class (correct order):")
print(f"  Accuracy:  {true_correct}/{true_total} = {true_correct/true_total:.2%}")
print(f"  Precision: {precision_per_class[1]:.4f}")
print(f"  Recall:    {recall_per_class[1]:.4f}")
print(f"  F1 Score:  {f1_per_class[1]:.4f}")

# 'false' class
false_total = sum(1 for gt in ground_truths if gt == 'false')
false_correct = sum(1 for gt, pred in zip(ground_truths, predictions) if gt == 'false' and pred == 'false')
print(f"\n'false' class (shuffled order):")
print(f"  Accuracy:  {false_correct}/{false_total} = {false_correct/false_total:.2%}")
print(f"  Precision: {precision_per_class[0]:.4f}")
print(f"  Recall:    {recall_per_class[0]:.4f}")
print(f"  F1 Score:  {f1_per_class[0]:.4f}")

print(f"\n{'='*70}")
print(f"CONFUSION MATRIX")
print(f"{'='*70}")
print(f"\n                Predicted 'false'  Predicted 'true'")
print(f"Actual 'false'        {cm[0,0]:4d}              {cm[0,1]:4d}")
print(f"Actual 'true'         {cm[1,0]:4d}              {cm[1,1]:4d}")

print(f"\n{'='*70}")
print(f"CLASSIFICATION REPORT")
print(f"{'='*70}")
print(classification_report(y_true, y_pred, target_names=['false', 'true'], digits=4))

# Save results
pd.DataFrame(results).to_csv("recipeqa_test_results.csv", index=False)
print(f"\nResults saved to recipeqa_test_results.csv")