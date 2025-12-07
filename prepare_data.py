import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

PROMPT = """TASK: Determine temporal order of LEFT and RIGHT images. 
    STEP 1: Describe key differences between the images (look for: burnt/cut objects, people positions, sun/shadow positions, object states, etc.)
    STEP 2: Analyze which state came first chronologically.
    STEP 3: State your final answer. FORMAT: Write "ANSWER: first" if LEFT happened earlier, or "ANSWER: second" if RIGHT happened earlier. Then explain your reasoning in 1-2 sentences."""

def create_viscot_dataset(csv_path, image_dir, output_json_path):
    """
    Convert your CSV to Visual-CoT format
    
    Visual-CoT format:
    {
        "id": "unique_id",
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "answer"}
        ]
    }
    """
    # Read CSV (no header, columns: index, image_path, temporal_order)
    df = pd.read_csv(csv_path, header=None, names=['index', 'image_path', 'temporal_order'])
    
    viscot_data = []
    
    for idx, row in df.iterrows():
        # Construct image filename from index
        image_filename = f"{row['index']}.jpg"
        full_image_path = os.path.join(image_dir, image_filename)
        
        # Get ground truth
        answer = row['temporal_order'].strip().lower()
        
        if answer not in ['first', 'second']:
            print(f"Warning: Invalid answer '{answer}' for index {row['index']}")
            continue
        
        # Create Visual-CoT format conversation
        conversation = {
            "id": f"temporal_{row['index']:04d}",
            "image": full_image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": PROMPT
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        viscot_data.append(conversation)
    
    # Save
    with open(output_json_path, 'w') as f:
        json.dump(viscot_data, f, indent=2)
    
    print(f"Created {len(viscot_data)} samples")
    print(f"Saved to: {output_json_path}")
    
    return viscot_data

def split_dataset(csv_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split dataset into train/val/test"""
    df = pd.read_csv(csv_path, header=None, names=['index', 'image_path', 'temporal_order'])
    
    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=df['temporal_order']
    )
    
    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio/(val_ratio + test_ratio),
        random_state=random_state,
        stratify=temp_df['temporal_order']
    )
    
    # Save splits
    os.makedirs('data_splits', exist_ok=True)
    train_df.to_csv('data_splits/train.csv', index=False, header=False)
    val_df.to_csv('data_splits/val.csv', index=False, header=False)
    test_df.to_csv('data_splits/test.csv', index=False, header=False)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"\nTrain distribution:")
    print(train_df['temporal_order'].value_counts())
    
    return train_df, val_df, test_df

def main():
    # Configuration - UPDATE THESE PATHS
    CSV_PATH = "/workspace/Visual-CoT/predictions_temporal_vqa_improved_prompt_2.csv"
    IMAGE_DIR = "/workspace/data/temporal_concat"
    
    # Step 1: Split data
    print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(CSV_PATH)
    
    # Step 2: Create JSON files
    print("\nCreating Visual-CoT format JSON files...")
    
    create_viscot_dataset('data_splits/train.csv', IMAGE_DIR, 'temporal_train.json')
    create_viscot_dataset('data_splits/val.csv', IMAGE_DIR, 'temporal_val.json')
    create_viscot_dataset('data_splits/test.csv', IMAGE_DIR, 'temporal_test.json')
    
    print("\n✓ Data preparation complete!")

if __name__ == "__main__":
    main()