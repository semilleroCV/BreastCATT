import argparse
import pandas as pd
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
        default="metadata_with_views.csv",
        help="Path to the local metadata file with the NEW prompts. Must be in the same order as the Hub dataset."
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
        default="with_full_metadata",
        help="The name for the new dataset configuration on the Hub."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    data_root_path = Path(args.local_data_path)

    # 1. Load the base dataset from the Hub to get embeddings and segmentations
    print(f"Loading base dataset '{args.dataset_repo_id}' with config '{args.base_config_name}'...")
    hub_dataset = load_dataset(args.dataset_repo_id, name=args.base_config_name, split="all")

    # 2. Read the local metadata CSV with the NEW prompts
    print(f"Reading local metadata with new prompts from '{args.metadata_csv_path}'...")
    local_df = pd.read_csv(args.metadata_csv_path).fillna('')

    # 3. Verify that the datasets have the same length, as we rely on their order corresponding.
    if len(hub_dataset) != len(local_df):
        print(f"\nError: Mismatch in lengths. Hub dataset has {len(hub_dataset)} entries, "
              f"but local CSV has {len(local_df)} entries.")
        print("The script requires these to be identical and in the same order.")
        return

    # 4. Process the data, combining Hub data with new local metadata row-by-row
    train_records, val_records, test_records = [], [], []
    print("Processing data, assuming a 1-to-1 correspondence in order...")
    
    # Convert DataFrame to a list of dictionaries for faster access
    local_records = local_df.to_dict('records')

    for i, hub_item in enumerate(tqdm(hub_dataset)):
        local_row = local_records[i]
        
        file_name = local_row['file_name']
        
        # The segmentation mask from the hub is already a list of lists.
        mask_list = hub_item['segmentation_mask']
        
        record = {
            "image": str(data_root_path / file_name),
            "label": local_row['label'],
            "text": local_row['text'],  # Using the NEW text from the local file
            "patient_id": local_row['patient_id'],
            "text_embedding": hub_item["text_embedding"],
            "segmentation_mask": mask_list,
            "protocol": local_row['protocol'],
            "view": local_row['view'],
        }

        if local_row['split'] == 'train':
            train_records.append(record)
        elif local_row['split'] == 'validation':
            val_records.append(record)
        else: # 'test'
            test_records.append(record)

    # 5. Define the features for the new dataset (WITH new columns)
    features = Features({
        'image': HFImage(),
        'label': ClassLabel(names=['benign', 'malignant']),
        'text': Value('string'),
        'text_embedding': Sequence(Value('float32')),
        'segmentation_mask': Sequence(Sequence(Sequence(Value('uint8')))),
        'patient_id': Value('string'),
        'protocol': Value('string'),
        'view': ClassLabel(names=['Frontal', 'Right 45°', 'Right 90°', 'Left 45°', 'Left 90°', 'Unknown']),
    })

    # 6. Create the final DatasetDict, only including non-empty splits
    final_dataset_dict = {}
    if train_records:
        final_dataset_dict["train"] = Dataset.from_list(train_records, features=features)
    if val_records:
        final_dataset_dict["validation"] = Dataset.from_list(val_records, features=features)
    if test_records:
        final_dataset_dict["test"] = Dataset.from_list(test_records, features=features)

    if not final_dataset_dict:
        print("\nError: No records were created. Check if the 'file_name' column in your CSV matches the filenames in the base dataset.")
        return

    final_dataset = DatasetDict(final_dataset_dict)

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