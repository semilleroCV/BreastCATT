import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel, Value, Image as HFImage, Sequence
from tqdm.auto import tqdm

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a full dataset config for DMR-IR from enriched metadata and upload to Hugging Face."
    )
    parser.add_argument(
        "--metadata_csv_path",
        type=str,
        default="dmr_ir_complete_metadata.csv",
        help="Path to the local metadata file that includes all clinical data."
    )
    parser.add_argument(
        "--local_data_path",
        type=str,
        required=True,
        help="Path to the root directory of the dataset, where the image files are stored (e.g., 'DMR-IR/')."
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
        help="The base config to get embeddings and segmentations from."
    )
    parser.add_argument(
        "--new_config_name",
        type=str,
        default="with_full_clinical_data",
        help="The name for the new dataset configuration on the Hub."
    )
    return parser.parse_args()

def get_hf_dtype(dtype):
    """Maps pandas dtype to Hugging Face Datasets Value type."""
    if pd.api.types.is_integer_dtype(dtype):
        return Value('int64')
    elif pd.api.types.is_float_dtype(dtype):
        return Value('float64')
    elif pd.api.types.is_bool_dtype(dtype):
        return Value('bool')
    else:
        return Value('string')

def main():
    args = parse_args()
    data_root_path = Path(args.local_data_path)

    # 1. Load the base dataset from the Hub to get embeddings and segmentations
    print(f"Loading base dataset '{args.dataset_repo_id}' with config '{args.base_config_name}'...")
    hub_dataset = load_dataset(args.dataset_repo_id, name=args.base_config_name, split="all")

    # 2. Create a lookup dictionary from the Hub dataset using 'text' as the key.
    metadata_lookup = {}
    print("Creating metadata lookup from the Hub dataset...")
    for item in tqdm(hub_dataset):
        metadata_lookup[item['text']] = {
            "text_embedding": item["text_embedding"],
            "segmentation_mask": item["segmentation_mask"],
        }
    print(f"Created metadata lookup with {len(metadata_lookup)} entries.")

    # 3. Read the local ENRICHED metadata CSV
    print(f"Reading local metadata from '{args.metadata_csv_path}'...")
    local_df = pd.read_csv(args.metadata_csv_path)
    
    # Define columns to be excluded from the final dataset features
    EXCLUDE_COLS = ['file_name', 'text', 'label', 'patient_id', 'split', 'Unnamed: 0', 'Unnamed: 0.1']

    # 4. Process the DataFrame to create new records and sort them into splits
    train_records, val_records, test_records = [], [], []
    print("Processing CSV, matching with Hub metadata, and assigning to splits...")
    for _, row in tqdm(local_df.iterrows(), total=len(local_df)):
        text = row['text']
        
        hub_meta = metadata_lookup.get(text)
        if hub_meta:
            mask_array = np.array(hub_meta['segmentation_mask'])
            if mask_array.ndim == 2:
                mask_array = np.expand_dims(mask_array, axis=0)
            
            record = {
                "image": str(data_root_path / row['file_name']),
                "label": row['label'],
                "text": text,
                "patient_id": row['patient_id'],
                "text_embedding": hub_meta["text_embedding"],
                "segmentation_mask": mask_array.tolist(),
            }
            
            # Add all other columns from the CSV to the record
            for col in local_df.columns:
                if col not in EXCLUDE_COLS and col not in record:
                    # Convert numpy types to native Python types for serialization
                    value = row[col]
                    if isinstance(value, np.generic):
                        value = value.item()
                    record[col] = value

            if row['split'] == 'train':
                train_records.append(record)
            elif row['split'] == 'validation':
                val_records.append(record)
            else: # 'test'
                test_records.append(record)

    # 5. Define the features for the new dataset dynamically
    features_dict = {
        'image': HFImage(),
        'label': ClassLabel(names=['benign', 'malignant']),
        'text': Value('string'),
        'text_embedding': Sequence(Value('float32')),
        'segmentation_mask': Sequence(Sequence(Sequence(Value('uint8')))),
        'patient_id': Value('string'),
    }
    
    # Add features for all other columns dynamically
    for col in local_df.columns:
        if col not in EXCLUDE_COLS and col not in features_dict:
            if col == 'view': # Special handling for 'view' to be a ClassLabel
                unique_views = local_df[col].unique().tolist()
                features_dict[col] = ClassLabel(names=unique_views)
            else:
                features_dict[col] = get_hf_dtype(local_df[col].dtype)

    features = Features(features_dict)

    # 6. Create the final DatasetDict
    final_dataset = DatasetDict({
        "train": Dataset.from_list(train_records, features=features),
        "validation": Dataset.from_list(val_records, features=features),
        "test": Dataset.from_list(test_records, features=features),
    })

    print("\nNew dataset structure created:")
    print(final_dataset)

    # 7. Push the new configuration to the Hub
    print(f"\nPushing new dataset to '{args.dataset_repo_id}' with config name '{args.new_config_name}'...")
    final_dataset.push_to_hub(
        args.dataset_repo_id,
        config_name=args.new_config_name,
    )
    print("\nDone! The new configuration has been successfully uploaded.")

if __name__ == "__main__":
    main()