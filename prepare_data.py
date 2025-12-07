import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

PROMPT = "Look at the two frames side-by-side. Which frame occurred first in time: the LEFT frame or the RIGHT frame? Answer with 'first' if the left frame came first, or 'second' if the right frame came first."

def create_viscot_dataset(csv_path, image_dir, output_json_path):
    """Convert CSV to Visual-CoT format"""
    df = pd.read_csv(csv_path, header=None, names=['index', 'image_path', 'temporal_order'])
    
    viscot_data = []
    
    for idx, row in df.iterrows():
        image_filename = f"{row['index']}.jpg"
        full_image_path = os.path.join(image_dir, image_filename)
        
        answer = row['temporal_order'].strip().lower()
        
        if answer not in ['first', 'second']:
            print(f"Warning: Invalid answer '{answer}' for index {row['index']}")
            continue
        
        conversation = {
            "id": f"temporal_{row['index']:04d}",
            "image": full_image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{PROMPT}"  # ← FIXED: Added <image> token
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        viscot_data.append(conversation)
    
    with open(output_json_path, 'w') as f:
        json.dump(viscot_data, f, indent=2)
    
    print(f"Created {len(viscot_data)} samples")
    print(f"Saved to: {output_json_path}")
    
    return viscot_data

def split_dataset(csv_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split dataset into train/val/test"""
    df = pd.read_csv(csv_path, header=None, names=['index', 'image_path', 'temporal_order'])
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=df['temporal_order']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio/(val_ratio + test_ratio),
        random_state=random_state,
        stratify=temp_df['temporal_order']
    )
    
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
    CSV_PATH = "/ocean/projects/cis250266p/kanand/VisCoT/predictions_temporal_vqa_improved_prompt_2.csv"
    IMAGE_DIR = "/ocean/projects/cis250266p/kanand/data/temporal_concat"
    
    print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(CSV_PATH)
    
    print("\nCreating Visual-CoT format JSON files...")
    create_viscot_dataset('data_splits/train.csv', IMAGE_DIR, 'temporal_train.json')
    create_viscot_dataset('data_splits/val.csv', IMAGE_DIR, 'temporal_val.json')
    create_viscot_dataset('data_splits/test.csv', IMAGE_DIR, 'temporal_test.json')
    
    print("\n✓ Data preparation complete!")

if __name__ == "__main__":
    main()