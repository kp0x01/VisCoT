#!/usr/bin/env python
import torch
# Disable flash attention
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import argparse

def run_inference(model_path, image_path, query, temperature=0.2, max_tokens=512):
    print(f"Loading model from {model_path}...")
    
    model_name = "llava-v1.5-7b"
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False
    )
    model = model.to(dtype=torch.float16)
    print(f"Model loaded! Context length: {context_len}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Processing image: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    conv = conv_templates["vicuna_v1"].copy()
    
    inp = DEFAULT_IMAGE_TOKEN + '\n' + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(f"\nQuery: {query}")
    print("Generating response...\n")
    
    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=max_tokens,
            use_cache=True,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Filter out invalid token IDs
    vocab_size = len(tokenizer)
    valid_output_ids = []
    for ids in output_ids:
        valid_ids = [id for id in ids.tolist() if 0 <= id < vocab_size]
        valid_output_ids.append(torch.tensor(valid_ids))
    
    # Decode with error handling
    try:
        outputs = tokenizer.batch_decode(valid_output_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        print(f"Decoding error: {e}")
        # Try decoding without skipping special tokens
        outputs = tokenizer.batch_decode(valid_output_ids, skip_special_tokens=False)[0].strip()
    
    return outputs

def main():
    parser = argparse.ArgumentParser(description='Visual-CoT Inference')
    parser.add_argument('--model-path', type=str, 
                       default='checkpoints/VisCoT-7b-336',
                       help='Path to the model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--query', type=str, required=True,
                       help='Question to ask about the image')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    result = run_inference(
        args.model_path,
        args.image,
        args.query,
        args.temperature,
        args.max_tokens
    )
    
    print("="*80)
    print("RESPONSE:")
    print("="*80)
    print(result)
    print("="*80)

if __name__ == "__main__":
    main()