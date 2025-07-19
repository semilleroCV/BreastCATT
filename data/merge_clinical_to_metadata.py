import pandas as pd
import argparse
from tqdm import tqdm

def merge_clinical_data(metadata_path, clinical_path, output_path):
    """
    Merges detailed clinical data into a metadata file based on patient_id.

    Args:
        metadata_path (str): Path to the CSV file with image metadata.
        clinical_path (str): Path to the CSV file with detailed clinical data.
        output_path (str): Path to save the merged CSV file.
    """
    # --- 1. Load the datasets ---
    print(f"Loading image metadata from: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    
    print(f"Loading clinical data from: {clinical_path}")
    clinical_df = pd.read_csv(clinical_path)

    # --- 2. Prepare for Merge ---
    # Normalize 'patient_id' in the metadata dataframe to serve as the merge key.
    # It handles formats like 'PAC_123' and ensures it's an integer.
    def normalize_id(pid):
        if isinstance(pid, str) and pid.startswith('PAC_'):
            return int(pid.replace('PAC_', ''))
        try:
            return int(pid)
        except (ValueError, TypeError):
            return None
            
    metadata_df['merge_id'] = metadata_df['patient_id'].apply(normalize_id)

    # The clinical data has one row per patient. We'll use its 'patient_id' as the key.
    # We rename it to 'merge_id' to perform a clean merge.
    clinical_to_merge = clinical_df.rename(columns={'patient_id': 'merge_id'})

    # --- 3. Perform the Merge ---
    print("Merging clinical data into metadata...")
    # A left merge keeps every record from the metadata (every image) and adds
    # the corresponding clinical data.
    merged_df = pd.merge(metadata_df, clinical_to_merge, on='merge_id', how='left')

    # --- 4. Clean Up and Verify ---
    # Replace the original patient_id with the normalized version.
    if 'merge_id' in merged_df.columns:
        merged_df['patient_id'] = merged_df['merge_id']
        merged_df.drop(columns=['merge_id'], inplace=True)

    # Drop the redundant 'diagnosis' column, as 'label' is already the integer version.
    if 'diagnosis' in merged_df.columns:
        merged_df.drop(columns=['diagnosis'], inplace=True)
        print("Dropped redundant 'diagnosis' column.")

    # Report on the success of the merge
    # We check one of the new columns to see how many rows got clinical data.
    if not clinical_df.empty:
        first_clinical_col = clinical_df.columns[1] # Use a column we know is from the clinical df
        matched_count = merged_df[first_clinical_col].notna().sum()
        total_count = len(merged_df)
        print(f"Successfully matched clinical data for {matched_count} of {total_count} image records.")
    else:
        print("Warning: Clinical data file is empty.")

    # --- 5. Save the Final Dataset ---
    print(f"Saving the complete dataset to: {output_path}")
    merged_df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge full clinical data into the main metadata CSV for the Hugging Face dataset.")
    parser.add_argument(
        "--metadata_csv", 
        type=str, 
        default="/home/guillermo/ssd/Github/BreastCATT/data/DMR-IR/with_full_metadata.csv", 
        help="Path to the metadata CSV with image records."
    )
    parser.add_argument(
        "--clinical_csv", 
        type=str, 
        default="/home/guillermo/ssd/Github/BreastCATT/data/dmr_ir_clinical_data.csv", 
        help="Path to the detailed clinical data CSV."
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="/home/guillermo/ssd/Github/BreastCATT/data/dmr_ir_complete_metadata.csv", 
        help="Path for the final, merged CSV file."
    )
    
    args = parser.parse_args()
    
    merge_clinical_data(args.metadata_csv, args.clinical_csv, args.output_csv)
