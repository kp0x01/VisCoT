# inference_finetuned.py
import torch
from transformers import AutoTokenizer
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
from peft import PeftModel

# SIMPLE PROMPT - matches training
SIMPLE_PROMPT = """Given two frames from a video, determine their temporal order. Answer with 'first' if the left frame occurs before the right frame. Answer with 'second' if the right frame occurs before the left frame."""

# EXPLANATION PROMPT - for second pass
EXPLANATION_PROMPT = """Look at these two frames. The left frame shows the earlier moment. Describe what visual cues indicate the temporal order (e.g., object states, positions, shadows, etc.)."""

print("Loading base model...")
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "checkpoints/VisCoT-7b-336",
    use_fast=False
)

model = LlavaLlamaForCausalLM.from_pretrained(
    "checkpoints/VisCoT-7b-336",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    model,
    "temporal_lora_output/checkpoint-80",
    is_trainable=False
)

print("Loading vision tower...")
vision_tower = model.get_vision_tower()
if vision_tower is None or not vision_tower.is_loaded:
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    vision_tower = CLIPVisionTower("openai/clip-vit-large-patch14-336")
    vision_tower.load_model()
    model.get_model().vision_tower = vision_tower

vision_tower.to(device='cuda', dtype=torch.float16)
image_processor = vision_tower.image_processor

model.eval()

print("Model loaded! Testing...")

def predict_with_explanation(image_path):
    """Two-stage: (1) Get answer, (2) Get explanation"""
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # ===== STAGE 1: Get the answer =====
    conv = conv_templates["vicuna_v1"].copy()
    inp = DEFAULT_IMAGE_TOKEN + '\n' + SIMPLE_PROMPT
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    print("Getting temporal order prediction...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=10,  # Short answer
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_token_len = input_ids.shape[1]
    answer = tokenizer.batch_decode(
        output_ids[:, input_token_len:], 
        skip_special_tokens=True
    )[0].strip().lower()
    
    # Extract just "first" or "second"
    if "first" in answer:
        temporal_answer = "first"
    elif "second" in answer:
        temporal_answer = "second"
    else:
        temporal_answer = answer
    
    print(f"Answer: {temporal_answer}")
    
    # ===== STAGE 2: Get explanation from base model =====
    # Use base Visual-CoT without LoRA for explanation
    print("Getting explanation...")
    
    # Adjust prompt based on answer
    if temporal_answer == "first":
        explanation_prompt = f"{EXPLANATION_PROMPT}"
    else:
        explanation_prompt = "Look at these two frames. The right frame shows the earlier moment. Describe what visual cues indicate the temporal order."
    
    conv2 = conv_templates["vicuna_v1"].copy()
    inp2 = DEFAULT_IMAGE_TOKEN + '\n' + explanation_prompt
    conv2.append_message(conv2.roles[0], inp2)
    conv2.append_message(conv2.roles[1], None)
    prompt2 = conv2.get_prompt()
    
    input_ids2 = tokenizer_image_token(
        prompt2, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    with torch.inference_mode():
        output_ids2 = model.generate(
            input_ids=input_ids2,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=200,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_token_len2 = input_ids2.shape[1]
    explanation = tokenizer.batch_decode(
        output_ids2[:, input_token_len2:], 
        skip_special_tokens=True
    )[0].strip()
    
    return temporal_answer, explanation

# Test
test_image = "/ocean/projects/cis250266p/kanand/data/temporal_concat/504.jpg"
answer, explanation = predict_with_explanation(test_image)

print(f"\n{'='*60}")
print(f"Test image: {test_image}")
print(f"ANSWER: {answer}")
print(f"\nEXPLANATION:")
print(explanation)
print(f"{'='*60}")