import os
import json
import re
import csv
import argparse
from pathlib import Path

from breastcatt.single_prompt import convert_json_to_sentence

def get_patient_id(filename):
    """
    Extract the patient ID from the filename.
    Expected formats: "PAC_1_001.tiff" or "PAC_1_001.txt"
    """
    match = re.match(r"PAC_(\d+)_.*\.(txt|tiff)$", filename, flags=re.IGNORECASE)
    return match.group(1) if match else None

def update_prompts_metadata(dataset_root, prompts_dir, output_csv):
    """
    Iterates over the dataset splits ("train" and "test") and classes ("benign" and "malignant").
    For each image file in .tiff format, retrieves its corresponding prompt text using the JSON file
    in prompts_dir, and writes the results into an output CSV with columns "file_name", "label", and "text".
    The file_name is stored as a relative path from dataset_root.
    
    Labels are assigned as:
      benign    -> 0
      malignant -> 1
    """
    updated_records = []
    splits = ["train", "test"]
    classes = ["benign", "malignant"]
    label_mapping = {"benign": 0, "malignant": 1}  # adjust if needed
    dataset_root_path = Path(dataset_root).resolve()
    
    for split in splits:
        for cls in classes:
            folder = dataset_root_path / split / cls
            if not folder.exists():
                print(f"Warning: {folder} does not exist.")
                continue
            # Iterate through .tiff files in the directory
            for file_path in folder.glob("*.tiff"):
                filename = file_path.name
                patient_id = get_patient_id(filename)
                if not patient_id:
                    print(f"Skipping file {filename}: patient ID not found.")
                    continue
                prompt_file = Path(prompts_dir) / f"{patient_id}.json"
                if not prompt_file.exists():
                    print(f"Warning: Prompt file {prompt_file} not found for patient {patient_id}. Skipping.")
                    continue
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt_data = json.load(f)
                text = convert_json_to_sentence(prompt_data)
                # Get the file's relative path from the dataset root (using '/' as separator)
                rel_path = file_path.relative_to(dataset_root_path).as_posix()
                record = {
                    "file_name": rel_path,
                    "label": label_mapping.get(cls, -1),
                    "text": text
                }
                updated_records.append(record)
    
    # Write collected records to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["file_name", "label", "text"])
        for record in updated_records:
            writer.writerow([record["file_name"], record["label"], record["text"]])
    
    print(f"Updated metadata CSV written to {output_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Update metadata CSV with prompts for each image using prompt JSON files."
    )
    parser.add_argument("dataset_root", type=str, help="Path to the dataset root directory (containing train and test folders).")
    parser.add_argument("prompts_dir", type=str, help="Path to the prompts directory (containing JSON files).")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path. Defaults to 'metadata.csv' in the dataset root.")
    
    args = parser.parse_args()
    output_csv = args.output if args.output else os.path.join(args.dataset_root, "metadata.csv")
    update_prompts_metadata(args.dataset_root, args.prompts_dir, output_csv)

if __name__ == "__main__":
    main()