import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset, Features, ClassLabel, Value, Image, Sequence

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Create a view-specific dataset config and upload to Hugging Face.")
    parser.add_argument(
        "--local_data_path",
        type=str,
        required=True,
        help="Path to the local directory with view-specific data (e.g., 'DMR-IR-VIEW')."
    )
    parser.add_argument(
        "--metadata_csv_path",
        type=str,
        required=True,
        help="Path to the local metadata.csv file."
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="SemilleroCV/DMR-IR",
        help="The repository ID of the dataset on the Hugging Face Hub."
    )
    parser.add_argument(
        "--base_config_name",
        type=str,
        default="with_embeddings_and_segmentation",
        help="The base config to get metadata from (e.g., text, embeddings)."
    )
    parser.add_argument(
        "--new_config_name",
        type=str,
        default="with_embeddings_and_segmentation_per_view",
        help="The name for the new dataset configuration on the Hub."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    local_data_path = Path(args.local_data_path)

    # 1. Load the original dataset and local metadata CSV
    print(f"Loading base dataset '{args.dataset_repo_id}' with config '{args.base_config_name}'...")
    original_dataset = load_dataset(args.dataset_repo_id, name=args.base_config_name, split="test")
    
    print(f"Loading local metadata from '{args.metadata_csv_path}'...")
    # Assuming the CSV has columns: file_name, label, text
    metadata_df = pd.read_csv(args.metadata_csv_path)
    metadata_df = metadata_df[metadata_df['file_name'].str.startswith('test/')]
    metadata_df['filename'] = metadata_df['file_name'].apply(lambda p: os.path.basename(p))
    
    # Create a lookup from text -> patient_id. This handles the duplicate text issue.
    metadata_df['patient_id'] = metadata_df['filename'].apply(lambda p: '_'.join(p.split('_')[:2]))
    text_to_patient_id = pd.Series(metadata_df.patient_id.values, index=metadata_df.text).to_dict()
    
    # Create the final lookup: patient_id -> {metadata from Hub}
    metadata_lookup = {}
    for item in original_dataset:
        patient_id = text_to_patient_id.get(item['text']) # type: ignore
        if patient_id and patient_id not in metadata_lookup:
            metadata_lookup[patient_id] = {
                'text': item['text'], # type: ignore
                'text_embedding': item['text_embedding'], # type: ignore
                'segmentation_mask': item['segmentation_mask'] # type: ignore
            }
    
    if not metadata_lookup:
        print("Error: Could not create metadata lookup. Check if text in metadata.csv matches text in the Hub dataset.")
        return

    print(f"Created metadata lookup for {len(metadata_lookup)} patients.")

    # 2. Walk the local directory to build the new dataset records
    records = []
    label_mapping = {"benign": 0, "malignant": 1}
    
    print(f"Processing local files from '{local_data_path}'...")
    for class_dir in local_data_path.joinpath("test").iterdir():
        if not class_dir.is_dir(): continue
        label_name = class_dir.name
        label = label_mapping[label_name]
        
        for view_dir in class_dir.iterdir():
            if not view_dir.is_dir(): continue
            view_name = view_dir.name
            
            for image_path in view_dir.glob("*.tiff"):
                image_filename = image_path.name
                patient_id = '_'.join(image_filename.split('_')[:2])
                
                if patient_id in metadata_lookup:
                    metadata = metadata_lookup[patient_id]
                    mask_array = np.array(metadata['segmentation_mask'])
                    records.append({
                        "image": str(image_path),
                        "label": label,
                        "view": view_name,
                        "text": metadata['text'],
                        "text_embedding": metadata['text_embedding'],
                        "segmentation_mask": mask_array
                    })
                else:
                    print(f"Warning: Metadata for {image_filename} (Patient ID: {patient_id}) not found. Skipping.")

    if not records:
        print("No records were created. Please check your local data path and the base dataset.")
        return

    # 3. Create a Hugging Face Dataset object directly from a dictionary of lists
    new_dataset_dict = {key: [dic[key] for dic in records] for key in records[0]}

    features = Features({
        'image': Image(),
        'label': ClassLabel(names=['benign', 'malignant']),
        'view': Value('string'),
        'text': Value('string'),
        'text_embedding': Sequence(Value('float32')),
        'segmentation_mask': Sequence(Sequence(Sequence(Value('uint8')))),
    })

    new_dataset = Dataset.from_dict(new_dataset_dict, features=features)

    print(f"\nCreated new dataset with {len(new_dataset)} records.")
    print("Example record:", new_dataset[0])

    # 4. Push the new dataset to the Hub
    print(f"\nPushing new dataset to '{args.dataset_repo_id}' with config name '{args.new_config_name}'...")
    new_dataset.push_to_hub(args.dataset_repo_id, config_name=args.new_config_name, split="test")
    
    print("\nDone! The 'per_view' configuration has been successfully uploaded.")

if __name__ == "__main__":
    main()