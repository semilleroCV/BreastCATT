import pandas as pd
import os
import re
from tqdm.auto import tqdm
import argparse

def parse_image_index_from_filename(filename):
    """
    Parses a filename like 'PAC_17_023.tiff' to extract ONLY the image index.
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r'_(\d+)$', base_name)
    if match:
        return int(match.group(1))
    return None

def parse_patient_id_from_column(patient_id_str):
    """
    Parses a patient_id string like 'PAC_17' to extract ONLY the numeric ID.
    """
    if not isinstance(patient_id_str, str):
        return None
    match = re.search(r'(\d+)$', patient_id_str)
    if match:
        return int(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Merge two CVS, one with scraped data from DMR-IR page and the metadata without views and protocol columns"
    )
    parser.add_argument("--output", type=str, default="metadata_with_views.csv", help="Output CSV file path.")
    parser.add_argument("--scraped_csv", type=str, required=True, help="Scraped CSV file path.")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Metadata CSV file path.")
    args = parser.parse_args()
    # --- Configuration ---
    scraped_csv_path = args.scraped_csv
    metadata_csv_path = args.metadata_csv
    output_csv_path = args.output

    # 1. Load the scraped views data
    print(f"Loading scraped views from '{scraped_csv_path}'...")
    views_df = pd.read_csv(scraped_csv_path)

    # 2. Process the scraped data to create a lookup dictionary
    # The key will be a tuple: (patient_id, image_index)
    view_lookup = {}
    print("Processing scraped data into a lookup table...")
    for _, row in tqdm(views_df.iterrows(), total=len(views_df)):
        # Create the composite key
        patient_id = int(row['patient_id_scraped'])
        image_index = int(row['image_index'])
        key = (patient_id, image_index)

        # Split 'view' into 'protocol' and 'view_angle'
        view_parts = row['view'].split(' - ', 1)
        protocol = view_parts[0].strip()
        view_angle = view_parts[1].strip() if len(view_parts) > 1 else 'Unknown'
        
        view_lookup[key] = {'protocol': protocol, 'view': view_angle}

    # 3. Load the main metadata file
    print(f"\nLoading main metadata from '{metadata_csv_path}'...")
    metadata_df = pd.read_csv(metadata_csv_path)

    # 4. Match and add the new columns
    protocols = []
    views = []
    print("Matching metadata and adding new columns...")
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        # Use the dedicated parsing functions
        patient_id = parse_patient_id_from_column(row['patient_id'])
        image_index = parse_image_index_from_filename(row['file_name'])
        
        if patient_id is not None and image_index is not None:
            key = (patient_id, image_index)
            # Find the corresponding view data in our lookup
            view_data = view_lookup.get(key)
            if view_data:
                protocols.append(view_data['protocol'])
                views.append(view_data['view'])
            else:
                # Handle cases where no match is found
                protocols.append(None)
                views.append(None)
        else:
            protocols.append(None)
            views.append(None)

    # Add the new data as columns to the DataFrame
    metadata_df['protocol'] = protocols
    metadata_df['view'] = views

    # 5. Save the enriched metadata to a new CSV file
    metadata_df.to_csv(output_csv_path, index=False)

    print(f"\nSuccessfully merged data and saved to '{output_csv_path}'.")
    print("New columns 'protocol' and 'view' have been added.")
    print("\nSample of the new data:")
    # This will now work because 'patient_id' exists in the original CSV
    print(metadata_df[['file_name', 'patient_id', 'protocol', 'view']].head())
    
    # Report on how many rows were successfully matched
    matched_count = metadata_df['protocol'].notna().sum()
    print(f"\nMatched {matched_count} out of {len(metadata_df)} rows.")

if __name__ == "__main__":
    main()  