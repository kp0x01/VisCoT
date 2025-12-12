#!/usr/bin/env python
import os

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
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

class TemporalOrderingInference:
    def __init__(self, model_path):
        """Initialize the model once for batch processing"""
        print(f"Loading model from {model_path}...")
        
        model_name = "llava-v1.5-7b"
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False
        )
        
        # Convert entire model to same dtype
        self.model = self.model.to(dtype=torch.float16)
        
        print(f"Model loaded! Vocab size: {len(self.tokenizer)}")
    
    def infer(self, image_path, query):
        """Run inference on a single image pair"""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            
            # Prepare conversation
            conv = conv_templates["vicuna_v1"].copy()
            inp = DEFAULT_IMAGE_TOKEN + '\n' + query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(self.model.device)
            
            # Generate response
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=50,  # Reduced since you only need one word
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Filter out invalid token IDs before decoding
            vocab_size = len(self.tokenizer)
            filtered_ids = []
            
            for ids in output_ids:
                valid_ids = []
                for token_id in ids.tolist():
                    if 0 <= token_id < vocab_size:
                        valid_ids.append(token_id)
                filtered_ids.append(torch.tensor(valid_ids, device=self.model.device))
            
            outputs = self.tokenizer.batch_decode(filtered_ids, skip_special_tokens=True)[0].strip()
            response_lower = outputs.lower()
            return outputs, None
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            return None, error_msg

def process_dataset(model_path, data_dir, output_file, query, image_extensions=None):
    """Process all images in the data directory"""
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Get all image files
    data_path = Path(data_dir)
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_path.glob(f'*{ext}'))
        image_files.extend(data_path.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images in {data_dir}")
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {data_dir}")
        return
    
    # Initialize model
    inferencer = TemporalOrderingInference(model_path)
    
    results = []
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        response, error = inferencer.infer(str(image_path), query)
        if 'ANSWER:' in response:
            first_word= response.split('ASSISTANT:')[1][:20]
            if "first" in first_word.lower():
                ans="first"
            else:
                ans="second"
        else:
            ans="NA"
        result = {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'query': query,
            'prediction': ans,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(result)
        
        # Save intermediate results every 10 images
        if len(results) % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved checkpoint at {len(results)} images")
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    # Print statistics
    successful = sum(1 for r in results if r['error'] is None)
    failed = len(results) - successful
    
    # Count predictions
    first_count = sum(1 for r in results if r['prediction'] == 'first')
    second_count = sum(1 for r in results if r['prediction'] == 'second')
    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"Total images: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Predictions - 'first': {first_count}, 'second': {second_count}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Temporal Ordering Inference')
    parser.add_argument('--model-path', type=str, 
                       default='checkpoints/VisCoT-7b-336',
                       help='Path to the Visual-CoT model')
    parser.add_argument('--data-dir', type=str, 
                       default='/workspace/data/temporal_concat',
                       help='Directory containing image pairs')
    parser.add_argument('--output', type=str, 
                       default='temporal_ordering_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--query', type=str,
    default='TASK: Determine temporal order of LEFT and RIGHT images. STEP 1: Describe key differences between the images (look for: burnt/cut objects, people positions, subjects entering and leaving the scene, sun/shadow positions, object states, effects of gravity etc.). STEP 2: Analyze which state came first chronologically. STEP 3: State your final answer. FORMAT: Write "ANSWER: first" if LEFT happened earlier, or "ANSWER: second" if RIGHT happened earlier. Then explain your reasoning in 1-2 sentences.',
    help='Question to ask about temporal ordering')

    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        return
    
    process_dataset(args.model_path, args.data_dir, args.output, args.query)

if __name__ == "__main__":
    main()