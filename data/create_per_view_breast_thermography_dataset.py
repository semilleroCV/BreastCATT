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
        description="Create a view-specific dataset config from a metadata CSV and upload to Hugging Face."
    )
    parser.add_argument(
        "--metadata_csv_path",
        type=str,
        required=True,
        help="Path to the local metadata.csv file."
    )
    parser.add_argument(
        "--local_data_path",
        type=str,
        required=True,
        help="Path to the root directory of the dataset, where the image files are stored (e.g., 'BreastThermography/')."
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="SemilleroCV/BreastThermography",
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
        default="with_embeddings_and_segmentation_per_view",
        help="The name for the new dataset configuration on the Hub."
    )
    return parser.parse_args()

def get_view_from_filename(filename):
    """Extracts the view from the filename."""
    filename_lower = filename.lower()
    if "_anterior." in filename_lower:
        return "frontal"
    elif "_oblleft." in filename_lower:
        return "left"
    elif "_oblright." in filename_lower:
        return "right"
    return "unknown"

def main():
    args = parse_args()
    data_root_path = Path(args.local_data_path)

    # 1. Load the base dataset from the Hub to get embeddings and segmentations
    print(f"Loading base dataset '{args.dataset_repo_id}' with config '{args.base_config_name}'...")
    hub_dataset = load_dataset(args.dataset_repo_id, name=args.base_config_name)

    # 2. Create a lookup dictionary from the Hub dataset.
    # The text is unique per patient, so it's a reliable key.
    metadata_lookup = {}
    for split in hub_dataset.keys():
        for item in hub_dataset[split]:
            metadata_lookup[item['text']] = {
                "text_embedding": item["text_embedding"],
                "segmentation_mask": item["segmentation_mask"],
            }
    print(f"Created metadata lookup with {len(metadata_lookup)} entries.")

    # 3. Read the local metadata CSV
    print(f"Reading local metadata from '{args.metadata_csv_path}'...")
    local_df = pd.read_csv(args.metadata_csv_path)

    # 4. Process the DataFrame to create new records
    new_records = []
    print("Processing CSV and matching with Hub metadata...")
    for _, row in tqdm(local_df.iterrows(), total=len(local_df)):
        file_name = row['file_name']
        text = row['text']
        
        # Match with Hub data
        hub_meta = metadata_lookup.get(text)
        if hub_meta:
            mask_array = np.array(hub_meta['segmentation_mask'])
            new_records.append({
                "image": str(data_root_path / file_name),
                "label": row['label'],
                "text": text,
                "view": get_view_from_filename(os.path.basename(file_name)),
                "patient_id": os.path.basename(file_name).split('_')[0],
                "text_embedding": hub_meta["text_embedding"],
                "segmentation_mask": mask_array.tolist()
            })

    # 5. Create the final Hugging Face Dataset
    
    # Define the features for the new dataset
    features = Features({
        'image': HFImage(),
        'label': ClassLabel(names=['benign', 'malignant']),
        'text': Value('string'),
        'text_embedding': Sequence(Value('float32')),
        'segmentation_mask': Sequence(Sequence(Sequence(Value('uint8')))),
        'view': Value('string'),
        'patient_id': Value('string'),
    })

    # Separate records into train and test splits first
    train_records = [r for r in new_records if '/train/' in r['image']]
    test_records = [r for r in new_records if '/test/' in r['image']]

    # Convert list of dicts to dict of lists
    train_dict = {key: [dic[key] for dic in train_records] for key in train_records[0]}
    test_dict = {key: [dic[key] for dic in test_records] for key in test_records[0]}

    # Create Dataset objects directly from dictionaries
    final_dataset = DatasetDict({
        "train": Dataset.from_dict(train_dict, features=features),
        "test": Dataset.from_dict(test_dict, features=features),
    })

    print("\nNew dataset structure created:")
    print(final_dataset)
    print("\nExample from train split:")
    print(final_dataset["train"][0])

    # 6. Push the new configuration to the Hub
    print(f"\nPushing new dataset to '{args.dataset_repo_id}' with config name '{args.new_config_name}'...")
    final_dataset.push_to_hub(
        args.dataset_repo_id,
        config_name=args.new_config_name,
    )
    print("\nDone! The new configuration has been successfully uploaded.")

if __name__ == "__main__":
    main()