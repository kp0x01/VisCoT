# prepare_recipeqa_data.py
import os
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
RECIPEQA_DIR = "/path/to/recipeqa_final"  # Update this
TRAIN_DIR = os.path.join(RECIPEQA_DIR, "train")
TEST_DIR = os.path.join(RECIPEQA_DIR, "test")

PROMPT = "Look at these four video frames shown in sequence from left to right. Are they in the correct temporal order? Answer 'true' if they are in the correct chronological order, or 'false' if they are shuffled or out of order."

def parse_filename(filename):
    """
    Parse filename like '5_1_true.jpg'
    Returns: (video_id, permutation_id, label)
    """
    name = filename.replace('.jpg', '')
    parts = name.split('_')
    
    if len(parts) == 3:
        video_id = parts[0]
        perm_id = parts[1]
        label = parts[2]  # 'true' or 'false'
        return video_id, perm_id, label
    return None, None, None

def create_recipeqa_dataset(image_dir, output_json):
    """Create dataset from RecipeQA images"""
    
    data = []
    
    # Get all jpg files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    for image_file in image_files:
        video_id, perm_id, label = parse_filename(image_file)
        
        if video_id is None:
            print(f"Warning: Could not parse {image_file}")
            continue
        
        if label not in ['true', 'false']:
            print(f"Warning: Invalid label '{label}' in {image_file}")
            continue
        
        full_image_path = os.path.join(image_dir, image_file)
        
        conversation = {
            "id": f"recipeqa_{video_id}_{perm_id}",
            "image": full_image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{PROMPT}"
                },
                {
                    "from": "gpt",
                    "value": label
                }
            ]
        }
        
        data.append(conversation)
    
    # Save
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created {len(data)} samples")
    print(f"Saved to: {output_json}")
    
    # Print distribution
    true_count = sum(1 for d in data if d['conversations'][1]['value'] == 'true')
    false_count = len(data) - true_count
    print(f"\nLabel distribution:")
    print(f"  true:  {true_count} ({100*true_count/len(data):.1f}%)")
    print(f"  false: {false_count} ({100*false_count/len(data):.1f}%)")
    
    return data

def split_recipeqa_by_video(image_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split dataset by VIDEO ID to avoid data leakage
    (all permutations of same video should be in same split)
    """
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # Group by video ID
    video_groups = {}
    for image_file in image_files:
        video_id, perm_id, label = parse_filename(image_file)
        if video_id is not None:
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(image_file)
    
    print(f"\nFound {len(video_groups)} unique videos")
    print(f"Average {len(image_files)/len(video_groups):.1f} permutations per video")
    
    # Split video IDs
    video_ids = list(video_groups.keys())
    
    # First split: train vs (val+test)
    train_ids, temp_ids = train_test_split(
        video_ids,
        test_size=(val_ratio + test_ratio),
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_ratio/(val_ratio + test_ratio),
        random_state=random_state
    )
    
    # Collect images for each split
    train_images = []
    val_images = []
    test_images = []
    
    for video_id in train_ids:
        train_images.extend(video_groups[video_id])
    for video_id in val_ids:
        val_images.extend(video_groups[video_id])
    for video_id in test_ids:
        test_images.extend(video_groups[video_id])
    
    print(f"\nSplit by video ID:")
    print(f"  Train: {len(train_ids)} videos, {len(train_images)} images")
    print(f"  Val:   {len(val_ids)} videos, {len(val_images)} images")
    print(f"  Test:  {len(test_ids)} videos, {len(test_images)} images")
    
    return train_images, val_images, test_images

def main():
    # Update this path
    IMAGE_DIR = "/ocean/projects/cis250266p/kanand/data/recipeqa_final/train"  # or wherever your images are
    
    # Option 1: Use existing train/test split
    if os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR):
        print("Using existing train/test directories...")
        
        create_recipeqa_dataset(TRAIN_DIR, 'recipeqa_train.json')
        create_recipeqa_dataset(TEST_DIR, 'recipeqa_test.json')
        
    # Option 2: Create custom split from all images
    else:
        print("Creating custom split...")
        
        train_images, val_images, test_images = split_recipeqa_by_video(IMAGE_DIR)
        
        # Create separate directories (optional)
        os.makedirs('recipeqa_splits/train', exist_ok=True)
        os.makedirs('recipeqa_splits/val', exist_ok=True)
        os.makedirs('recipeqa_splits/test', exist_ok=True)
        
        # Write image lists to JSON
        def create_split_json(image_list, image_dir, output_json):
            data = []
            for image_file in image_list:
                video_id, perm_id, label = parse_filename(image_file)
                full_path = os.path.join(image_dir, image_file)
                
                data.append({
                    "id": f"recipeqa_{video_id}_{perm_id}",
                    "image": full_path,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{PROMPT}"},
                        {"from": "gpt", "value": label}
                    ]
                })
            
            with open(output_json, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Created {len(data)} samples in {output_json}")
        
        create_split_json(train_images, IMAGE_DIR, 'recipeqa_train.json')
        create_split_json(val_images, IMAGE_DIR, 'recipeqa_val.json')
        create_split_json(test_images, IMAGE_DIR, 'recipeqa_test.json')
    
    print("\n✓ RecipeQA data preparation complete!")

if __name__ == "__main__":
    main()