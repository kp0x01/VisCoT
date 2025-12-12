# evaluate_accuracy.py
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
    confusion_matrix, 
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score,
    accuracy_score
)
import pandas as pd

PROMPT = """Look at the two frames side-by-side. Which frame occurred first in time: the LEFT frame or the RIGHT frame? Answer with 'first' if the left frame came first, or 'second' if the right frame came first."""

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

# update as needed here
CHECKPOINT_PATH = "/ocean/projects/cis250266p/kanand/VisCoT/temporal_lora_fixed/checkpoint-29"
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
with open("temporal_test.json", 'r') as f:
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
    
    # Extract answer
    if 'first' in pred_raw:
        pred = 'first'
    elif 'second' in pred_raw:
        pred = 'second'
    else:
        pred = 'first'  # default - first
    
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
    
    # Show first few examples
    if len(results) <= 5:
        print(f"\nImage: {image_path}")
        print(f"GT: {gt}, Pred: {pred}, Correct: {is_correct}")

# Calculate metrics
accuracy = accuracy_score(ground_truths, predictions)
# For binary classification, we'll treat 'first' as positive class (1) and 'second' as negative class (0)
# Convert to binary labels
y_true = [1 if gt == 'first' else 0 for gt in ground_truths]
y_pred = [1 if pred == 'first' else 0 for pred in predictions]
# Calculate metrics for both classes
precision_macro = precision_score(y_true, y_pred, average='macro')
recall_macro = recall_score(y_true, y_pred, average='macro')
f1_macro = f1_score(y_true, y_pred, average='macro')

precision_weighted = precision_score(y_true, y_pred, average='weighted')
recall_weighted = recall_score(y_true, y_pred, average='weighted')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

# Per-class metrics
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)
f1_per_class = f1_score(y_true, y_pred, average=None)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print results
print(f"\n{'='*70}")
print(f"OVERALL METRICS")
print(f"{'='*70}")
print(f"Test Accuracy: {accuracy:.2%} ({int(accuracy*len(test_data))}/{len(test_data)})")
print(f"\nMacro Averages (treats both classes equally):")
print(f"  Precision: {precision_macro:.4f}")
print(f"  Recall:    {recall_macro:.4f}")
print(f"  F1 Score:  {f1_macro:.4f}")
print(f"\nWeighted Averages (accounts for class imbalance):")
print(f"  Precision: {precision_weighted:.4f}")
print(f"  Recall:    {recall_weighted:.4f}")
print(f"  F1 Score:  {f1_weighted:.4f}")

print(f"\n{'='*70}")
print(f"PER-CLASS METRICS")
print(f"{'='*70}")

# 'first' class (positive, index 1)
first_total = sum(1 for gt in ground_truths if gt == 'first')
first_correct = sum(1 for gt, pred in zip(ground_truths, predictions) if gt == 'first' and pred == 'first')
print(f"\n'first' class (predicted {sum(y_pred)}/{len(y_pred)} times):")
print(f"  Accuracy:  {first_correct}/{first_total} = {first_correct/first_total:.2%}")
print(f"  Precision: {precision_per_class[1]:.4f}")
print(f"  Recall:    {recall_per_class[1]:.4f}")
print(f"  F1 Score:  {f1_per_class[1]:.4f}")

# 'second' class (negative, index 0)
second_total = sum(1 for gt in ground_truths if gt == 'second')
second_correct = sum(1 for gt, pred in zip(ground_truths, predictions) if gt == 'second' and pred == 'second')
print(f"\n'second' class (predicted {len(y_pred) - sum(y_pred)}/{len(y_pred)} times):")
print(f"  Accuracy:  {second_correct}/{second_total} = {second_correct/second_total:.2%}")
print(f"  Precision: {precision_per_class[0]:.4f}")
print(f"  Recall:    {recall_per_class[0]:.4f}")
print(f"  F1 Score:  {f1_per_class[0]:.4f}")

print(f"\n{'='*70}")
print(f"CONFUSION MATRIX")
print(f"{'='*70}")
print(f"\nRows = Ground Truth, Columns = Prediction")
print(f"\n                Predicted 'second'  Predicted 'first'")
print(f"Actual 'second'        {cm[0,0]:4d}              {cm[0,1]:4d}")
print(f"Actual 'first'         {cm[1,0]:4d}              {cm[1,1]:4d}")

# Also print sklearn's classification report for detailed breakdown
print(f"\n{'='*70}")
print(f"CLASSIFICATION REPORT")
print(f"{'='*70}")
class_names = ['second', 'first']
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("test_results.csv", index=False)
print(f"\nDetailed results saved to test_results.csv")

# Save metrics summary
metrics_summary = {
    'accuracy': accuracy,
    'precision_macro': precision_macro,
    'recall_macro': recall_macro,
    'f1_macro': f1_macro,
    'precision_weighted': precision_weighted,
    'recall_weighted': recall_weighted,
    'f1_weighted': f1_weighted,
    'first_accuracy': first_correct/first_total,
    'first_precision': precision_per_class[1],
    'first_recall': recall_per_class[1],
    'first_f1': f1_per_class[1],
    'second_accuracy': second_correct/second_total,
    'second_precision': precision_per_class[0],
    'second_recall': recall_per_class[0],
    'second_f1': f1_per_class[0],
}

pd.DataFrame([metrics_summary]).to_csv("metrics_summary.csv", index=False)
print("Metrics summary saved to metrics_summary.csv")

print(f"\n{'='*70}")