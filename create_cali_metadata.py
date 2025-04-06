import os
import csv
import re
import argparse
from pathlib import Path

def convert_row_to_sentence(row):
    """
    Generate a descriptive sentence from a CSV metadata row.
    Expected row fields: "Age(years)", "Weight (Kg)", "Height(cm)", "Temp(°C)", "Left", "Right".
    
    The pathology codes for 'Left' and 'Right' are interpreted as follows:
      PM: Malignant pathology for carcinoma stages – if found, this side is ignored in the text.
      PB: Pathological findings such as collagenized stroma, fibroadenoma, cyst, adenosis, 
          apocrine metaplasia, stromal fibrosis, epithelial hyperplasia, microcalcifications or nodule.
      N: Normal stages.
      
    Returns a sentence summarizing the patient's demographic information and only the pathology
    findings for the breast(s) where the code is not PM.
    """
    age = row.get("Age(years)", "unknown")
    weight = row.get("Weight (Kg)", "unknown")
    height = row.get("Height(cm)", "unknown")
    temp = row.get("Temp(°C)", "unknown")
    left_code = row.get("Left", "unknown")
    right_code = row.get("Right", "unknown")
    
    # Mapping for pathology codes:
    pathology_mapping = {
        "PB": "findings such as collagenized stroma, fibroadenoma, cyst, adenosis, apocrine metaplasia, stromal fibrosis, epithelial hyperplasia, microcalcifications or nodule",
        "N": "normal stages"
    }
    
    # Build descriptions only for sides that are not PM.
    descriptions = []
    if left_code != "PM":
        # If left_code is either PB or N, use the descriptive mapping; otherwise just show what it is.
        left_description = pathology_mapping.get(left_code, left_code)
        descriptions.append(f"The left breast presents {left_description}")
    if right_code != "PM":
        right_description = pathology_mapping.get(right_code, right_code)
        descriptions.append(f"The right breast shows {right_description}")
    
    # Compose the final sentence.
    demographic_text = (
        f"Patient is {age} years old, weighs {weight} kilograms, and is {height} centimeters tall. "
        f"Regarding the protocol, a body temperature of {temp} degrees Celsius."
    )
    
    if descriptions:
        breasts_text = ". ".join(descriptions) + "."
        sentence = f"{demographic_text} {breasts_text}"
    else:
        sentence = demographic_text
    
    return sentence

def load_metadata_csv(metadata_csv_path):
    """
    Reads the metadata CSV and returns a dictionary keyed by the base file name.
    The CSV is expected to have a column "file_name" containing values like "IIR0001.tiff".
    """
    metadata = {}
    with open(metadata_csv_path, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row.get("file_name")
            if key:
                metadata[key] = row
    return metadata

def extract_base_filename(filename):
    """
    From an image filename like:
      "IIR0001_anterior.tiff", "IIR0001_oblleft.tiff", "IIR0001_oblright.tiff"
    extract the base filename "IIR0001.tiff".
    """
    match = re.match(r"^(IIR\d{4})_.*\.tiff$", filename, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}.tiff"
    return None

def update_secondary_metadata(dataset_root, metadata_csv, output_csv):
    """
    Iterates over both dataset splits ("train" and "test") and classes ("benign", "malignant").
    For each image file (which has an extended name) it extracts its base file name,
    finds its corresponding metadata row, converts that row to a descriptive sentence, and
    writes an output CSV with columns "file_name", "label", and "text".
    
    Labels are assigned as:
      benign    -> 0
      malignant -> 1
    """
    updated_records = []
    splits = ["train", "test"]
    classes = ["benign", "malignant"]
    label_mapping = {"benign": 0, "malignant": 1}
    dataset_root_path = Path(dataset_root).resolve()
    
    # Load metadata CSV into a dictionary keyed by base file name.
    metadata_dict = load_metadata_csv(metadata_csv)
    
    for split in splits:
        for cls in classes:
            folder = dataset_root_path / split / cls
            if not folder.exists():
                print(f"Warning: Folder {folder} does not exist.")
                continue
            # Iterate over all .tiff files in the folder
            for file_path in folder.glob("*.tiff"):
                filename = file_path.name
                base_filename = extract_base_filename(filename)
                if not base_filename:
                    print(f"Skipping file {filename}: could not extract base filename.")
                    continue
                if base_filename not in metadata_dict:
                    print(f"Warning: Metadata for {base_filename} not found. Skipping {filename}.")
                    continue
                row = metadata_dict[base_filename]
                text = convert_row_to_sentence(row)
                rel_path = file_path.relative_to(dataset_root_path).as_posix()
                record = {
                    "file_name": rel_path,
                    "label": label_mapping.get(cls, -1),
                    "text": text
                }
                updated_records.append(record)
    
    # Write the collected records into a new CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["file_name", "label", "text"])
        for record in updated_records:
            writer.writerow([record["file_name"], record["label"], record["text"]])
    
    print(f"Updated metadata CSV written to {output_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Update metadata CSV for the secondary dataset by generating descriptive text from CSV metadata."
    )
    parser.add_argument("dataset_root", help="Path to the dataset root directory (containing train and test folders).")
    parser.add_argument("metadata_csv", help="Path to the CSV metadata file (with base file names).")
    parser.add_argument("--output", help="Output CSV file path. Defaults to 'metadata_updated.csv' in the dataset root.", default=None)
    
    args = parser.parse_args()
    output_csv = args.output if args.output else os.path.join(args.dataset_root, "metadata_updated.csv")
    update_secondary_metadata(args.dataset_root, args.metadata_csv, output_csv)

if __name__ == "__main__":
    main()