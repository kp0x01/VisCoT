import torch
import transformers
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
import json
import os
from typing import Dict, List

# Model settings
BASE_MODEL = "llava-hf/llava-1.5-7b-hf"
OUTPUT_DIR = "./temporal_lora_output"

# Data paths
TRAIN_JSON = "temporal_train.json"
VAL_JSON = "temporal_val.json"

# LoRA hyperparameters
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Training hyperparameters
BATCH_SIZE = 4  # Adjust based on your GPU
GRAD_ACCUM_STEPS = 4  # Effective batch size = 4*4 = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 20  # More epochs for small dataset
MAX_LENGTH = 512
WARMUP_RATIO = 0.1

# Use 4-bit quantization (QLoRA)
USE_QLORA = True


PROMPT = """TASK: Determine temporal order of LEFT and RIGHT images. 
    STEP 1: Describe key differences between the images (look for: burnt/cut objects, people positions, sun/shadow positions, object states, etc.)
    STEP 2: Analyze which state came first chronologically.
    STEP 3: State your final answer. FORMAT: Write "ANSWER: first" if LEFT happened earlier, or "ANSWER: second" if RIGHT happened earlier. Then explain your reasoning in 1-2 sentences."""

class TemporalQADataset(torch.utils.data.Dataset):
    def __init__(self, json_path, processor, max_length=512):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        
        print(f"Loaded {len(self.data)} samples from {json_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load image
        try:
            image = Image.open(sample['image']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image']}: {e}")
            # Return a dummy black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get conversation
        conversations = sample['conversations']
        question = conversations[0]['value']
        answer = conversations[1]['value']
        
        # Format as LLaVA conversation
        # Question already contains <image> token
        text = f"USER: {question} ASSISTANT: {answer}"
        
        # Process inputs
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Create labels for language modeling
        labels = inputs["input_ids"].clone()
        
        # Mask out the question part (only compute loss on answer)
        # Find "ASSISTANT:" in the tokenized sequence
        assistant_str = "ASSISTANT:"
        assistant_tokens = self.processor.tokenizer.encode(
            assistant_str, 
            add_special_tokens=False
        )
        
        # Find where ASSISTANT: appears
        input_ids_list = inputs["input_ids"].tolist()
        try:
            # Find the position of ASSISTANT: tokens
            for i in range(len(input_ids_list) - len(assistant_tokens) + 1):
                if input_ids_list[i:i+len(assistant_tokens)] == assistant_tokens:
                    # Mask everything before and including "ASSISTANT:"
                    labels[:i+len(assistant_tokens)] = -100
                    break
        except:
            # If we can't find it, mask first half (fallback)
            labels[:len(labels)//2] = -100
        
        inputs["labels"] = labels
        
        return inputs

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    # Stack all tensors
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key in ['input_ids', 'attention_mask', 'labels']:
            collated[key] = torch.stack([item[key] for item in batch])
        elif key == 'pixel_values':
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated

# ============================================================================
# Model Loading
# ============================================================================

def load_model_with_lora():
    """Load base model and add LoRA adapters"""
    
    print(f"Loading base model: {BASE_MODEL}")
    
    # Quantization config for QLoRA
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
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        print("Using FP16 (no quantization)")
        model = LlavaForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # LoRA configuration
    print(f"Adding LoRA adapters (r={LORA_R}, alpha={LORA_ALPHA})")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            # Attention layers
            "q_proj", "k_proj", "v_proj", "o_proj",
            # MLP layers (for LLaMA/Vicuna)
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)\n")
    
    return model

# ============================================================================
# Training
# ============================================================================

def train():
    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.padding_side = "right"  # Important for training
    
    # Load model with LoRA
    model = load_model_with_lora()
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = TemporalQADataset(TRAIN_JSON, processor, MAX_LENGTH)
    val_dataset = TemporalQADataset(VAL_JSON, processor, MAX_LENGTH)
    
    # Training arguments
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
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Train
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")
    
    trainer.train()
    
    # Save final model
    final_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    print(f"\nSaving final model to {final_checkpoint_dir}")
    trainer.save_model(final_checkpoint_dir)
    processor.save_pretrained(final_checkpoint_dir)
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    print(f"\nModel saved to: {final_checkpoint_dir}")
    print(f"TensorBoard logs: {OUTPUT_DIR}")
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {OUTPUT_DIR}")

if __name__ == "__main__":
    train()