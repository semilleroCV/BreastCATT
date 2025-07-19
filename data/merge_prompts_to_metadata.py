import pandas as pd
import argparse
import os

def merge_prompts_to_metadata(prompts_csv_path, metadata_csv_path, output_csv_path, drop_duplicates):
    """
    Merges prompts into a metadata file based on patient_id.

    Args:
        prompts_csv_path (str): Path to the CSV file containing 'patient_id' and 'prompt'.
        metadata_csv_path (str): Path to the CSV file containing image metadata and 'patient_id'.
        output_csv_path (str): Path to save the merged CSV file.
    """
    # --- 1. Load the datasets ---
    print(f"Loading prompts from: {prompts_csv_path}")
    prompts_df = pd.read_csv(prompts_csv_path)
    
    print(f"Loading metadata from: {metadata_csv_path}")
    metadata_df = pd.read_csv(metadata_csv_path)

    # --- 2. Prepare and Normalize IDs ---
    # Ensure the prompts dataframe has a 'text' column for consistency
    if 'prompt' in prompts_df.columns and 'text' not in prompts_df.columns:
        prompts_df.rename(columns={'prompt': 'text'}, inplace=True)

    # Create a mapping from patient_id to prompt for efficient lookup.
    # Duplicates are dropped to ensure a clean 1-to-1 mapping.
    prompts_map = prompts_df.drop_duplicates(subset=['patient_id']).set_index('patient_id')['text']

    # Normalize patient_id in the metadata dataframe.
    # This function handles both string IDs (e.g., 'PAC_123') and numeric IDs.
    def normalize_id(pid):
        if isinstance(pid, str):
            # Remove prefix and convert to integer
            return int(pid.replace('PAC_', ''))
        # Return as is if already numeric
        return int(pid)

    # Create a temporary column with the normalized ID for mapping
    metadata_df['normalized_id'] = metadata_df['patient_id'].apply(normalize_id)

    # --- 3. Map Prompts to Metadata ---
    print("Matching prompts to metadata records...")
    # Use the normalized ID to map the prompts from the prompts_map
    metadata_df['text'] = metadata_df['normalized_id'].map(prompts_map)

    # --- 4. Clean up, Verify, and Save ---
    # Replace the old patient_id column with the new normalized one
    metadata_df['patient_id'] = metadata_df['normalized_id']
    metadata_df = metadata_df.drop(columns=['normalized_id'])
    
    # Check how many prompts were successfully matched
    matched_count = metadata_df['text'].notna().sum()
    total_count = len(metadata_df)
    print(f"Successfully matched {matched_count} of {total_count} records.")
    
    unmatched_ids = metadata_df[metadata_df['text'].isna()]['patient_id'].unique()
    if unmatched_ids.any():
        print(f"Could not find prompts for {len(unmatched_ids)} unique patient_ids.")

    # --- 5. Optional: Drop duplicates to keep one record per patient ---
    if drop_duplicates:
        print("Dropping duplicate rows to keep one record per patient...")
        metadata_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)
        print(f"Filtered down to {len(metadata_df)} unique patient records.")

    # --- 6. Reorder columns and Save ---
    if 'text' in metadata_df.columns:
        cols = metadata_df.columns.tolist()
        # Find other key columns if they exist to put them at front
        key_cols = ['file_name','label','text','patient_id','split','protocol','view']
        
        # Get the list of columns that exist in the dataframe
        existing_key_cols = [col for col in key_cols if col in cols]
        other_cols = [col for col in cols if col not in existing_key_cols]
        
        metadata_df = metadata_df[existing_key_cols + other_cols]

    # Save the updated dataframe
    print(f"Saving updated metadata to: {output_csv_path}")
    metadata_df.to_csv(output_csv_path, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge prompts from one CSV into a metadata CSV based on patient_id.")
    parser.add_argument(
        "--prompts_csv", 
        type=str, 
        default="data/prompt_embeddings.csv", 
        help="Path to the CSV file with patient IDs and prompts."
    )
    parser.add_argument(
        "--metadata_csv", 
        type=str, 
        default="data/dmr_ir_metadata_w_views.csv", 
        help="Path to the main metadata CSV with image records."
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="data/dmr_ir_metadata_w_prompts.csv", 
        help="Path for the output CSV file."
    )
    parser.add_argument("--drop_duplicates", action='store_true', help="Only generate and save one row per patient.")
    
    args = parser.parse_args()
    
    merge_prompts_to_metadata(args.prompts_csv, args.metadata_csv, args.output_csv, args.drop_duplicates)
