#!/usr/bin/env python
"""
Few-shot temporal ordering with actual example images
"""
import os
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image, ImageDraw, ImageFont
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

class FewShotTemporalInference:
    def __init__(self, model_path, example_dir=None):
        """Initialize model and load example images"""
        print(f"Loading model from {model_path}...")
        
        model_name = "llava-v1.5-7b"
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False
        )
        
        self.model = self.model.to(dtype=torch.float16)
        print(f"Model loaded! Vocab size: {len(self.tokenizer)}")
        
        # Load example images if provided
        self.example_images = []
        self.example_labels = []
        
        if example_dir:
            self.load_examples(example_dir)
    
    def load_examples(self, example_dir):
        """Load example images and their labels"""
        example_path = Path(example_dir)
        
        # Load examples.json which contains image paths and labels
        examples_file = example_path / "examples.json"
        if examples_file.exists():
            with open(examples_file, 'r') as f:
                examples_data = json.load(f)
            
            for ex in examples_data:
                img_path = example_path / ex['image']
                if img_path.exists():
                    self.example_images.append(str(img_path))
                    self.example_labels.append(ex['label'])
            
            print(f"Loaded {len(self.example_images)} example images")
        else:
            print(f"Warning: {examples_file} not found. No examples loaded.")
    
    def create_few_shot_image(self, query_image_path, num_shots=2):
        """
        Create a single image combining example images with query image
        
        Layout:
        [Example 1] [Label: first]
        [Example 2] [Label: second]
        [Query Image] [?]
        """
        # Load query image
        query_img = Image.open(query_image_path).convert('RGB')
        
        # Limit examples to num_shots
        example_imgs = []
        for img_path in self.example_images[:num_shots]:
            example_imgs.append(Image.open(img_path).convert('RGB'))
        
        if len(example_imgs) == 0:
            # No examples, just return query image
            return query_img
        
        # Calculate dimensions for concatenated image
        # All images should have similar width, stack vertically
        max_width = max([img.width for img in example_imgs + [query_img]])
        
        # Resize all images to same width while maintaining aspect ratio
        resized_imgs = []
        for img in example_imgs + [query_img]:
            aspect = img.height / img.width
            new_height = int(max_width * aspect)
            resized = img.resize((max_width, new_height), Image.LANCZOS)
            resized_imgs.append(resized)
        
        # Add space for text labels (50 pixels per label)
        label_height = 50
        total_height = sum(img.height + label_height for img in resized_imgs)
        
        # Create combined image
        combined = Image.new('RGB', (max_width, total_height), color='white')
        
        # Paste images and add labels
        y_offset = 0
        draw = ImageDraw.Draw(combined)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 36)
        except:
            font = ImageFont.load_default()
        
        # Paste example images with labels
        for i, (img, label) in enumerate(zip(resized_imgs[:-1], self.example_labels[:num_shots])):
            combined.paste(img, (0, y_offset))
            y_offset += img.height
            
            # Add label
            label_text = f"Example {i+1}: Answer = {label}"
            draw.text((10, y_offset + 10), label_text, fill='black', font=font)
            y_offset += label_height
        
        # Paste query image
        combined.paste(resized_imgs[-1], (0, y_offset))
        y_offset += resized_imgs[-1].height
        
        # Add query marker
        draw.text((10, y_offset + 10), "Query: Answer = ?", fill='red', font=font)
        
        return combined
    
    def infer(self, image_path, query, use_few_shot=False, num_shots=2):
        """Run inference with optional few-shot examples"""
        try:
            # Create few-shot image if requested
            if use_few_shot and len(self.example_images) > 0:
                image = self.create_few_shot_image(image_path, num_shots)
                
                # Enhanced query for few-shot
                enhanced_query = f"""Look at the images above. I've shown you {num_shots} example(s) with their correct answers.

Now answer this query image (marked with "?") using the same reasoning:
{query}"""
            else:
                # Regular zero-shot
                image = Image.open(image_path).convert('RGB')
                enhanced_query = query
            
            # Process image
            image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            
            # Prepare conversation
            conv = conv_templates["vicuna_v1"].copy()
            inp = DEFAULT_IMAGE_TOKEN + '\n' + enhanced_query
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
            
            # Generate
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=100,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Filter and decode
            vocab_size = len(self.tokenizer)
            filtered_ids = []
            for ids in output_ids:
                valid_ids = [tid for tid in ids.tolist() if 0 <= tid < vocab_size]
                filtered_ids.append(torch.tensor(valid_ids, device=self.model.device))
            
            outputs = self.tokenizer.batch_decode(filtered_ids, skip_special_tokens=True)[0].strip()
            
            # Extract answer
            response_lower = outputs.lower()
            # if "first" in response_lower:
            #     result = "first"
            # elif "second" in response_lower:
            #     result = "second"
            # else:
            result = outputs
            
            return result, None
            
        except Exception as e:
            import traceback
            return None, f"{str(e)}\n{traceback.format_exc()}"

def process_dataset(model_path, data_dir, output_file, query, example_dir=None, 
                   use_few_shot=False, num_shots=2):
    """Process dataset with few-shot learning"""
    
    data_path = Path(data_dir)
    image_files = sorted(list(data_path.glob('*.jpg')) + 
                        list(data_path.glob('*.png')) +
                        list(data_path.glob('*.jpeg')))
    
    print(f"Found {len(image_files)} images in {data_dir}")
    
    if len(image_files) == 0:
        print(f"ERROR: No images found")
        return
    
    # Initialize model with examples
    inferencer = FewShotTemporalInference(model_path, example_dir)
    
    results = []
    
    for image_path in tqdm(image_files, desc="Processing"):
        response, error = inferencer.infer(str(image_path), query, use_few_shot, num_shots)
        
        result = {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'query': query,
            'use_few_shot': use_few_shot,
            'num_shots': num_shots if use_few_shot else 0,
            'prediction': response,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(result)
        
        if len(results) % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Statistics
    successful = sum(1 for r in results if r['error'] is None)
    first_count = sum(1 for r in results if r['prediction'] == 'first')
    second_count = sum(1 for r in results if r['prediction'] == 'second')
    
    print(f"\n{'='*80}")
    print(f"Complete! Total: {len(results)} | Success: {successful}")
    print(f"Predictions - first: {first_count} | second: {second_count}")
    print(f"Results: {output_file}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Few-Shot Temporal Ordering with Images')
    parser.add_argument('--model-path', type=str, default='checkpoints/VisCoT-7b-336')
    parser.add_argument('--data-dir', type=str, default='/workspace/data/temporal_concat')
    parser.add_argument('--example-dir', type=str, default=None,
                       help='Directory containing example images and examples.json')
    parser.add_argument('--output', type=str, default='few_shot_results.json')
    parser.add_argument('--query', type=str,
                       default='TASK: Temporal ordering of two images (LEFT and RIGHT). STEP 1: Describe what changed between images. STEP 2: Determine which state came first chronologically. STEP 3: If LEFT is earlier state, answer "first". If RIGHT is earlier state, answer "second". FORMAT: ANSWER: [first or second] REASONING: [one sentence why]')
    parser.add_argument('--few-shot', action='store_true',
                       help='Use few-shot with example images')
    parser.add_argument('--num-shots', type=int, default=2,
                       help='Number of example images to include')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return
    
    process_dataset(args.model_path, args.data_dir, args.output, args.query,
                   args.example_dir, args.few_shot, args.num_shots)

if __name__ == "__main__":
    main()