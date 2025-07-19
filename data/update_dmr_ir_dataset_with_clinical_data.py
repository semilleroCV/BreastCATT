import argparse
import pandas as pd
from datasets import DatasetDict, Features, Value, ClassLabel, load_dataset

def update_dataset_with_clinical_data(
    repo_id,
    metadata_csv_path,
    base_config_name,
    new_config_name,
):
    """
    Updates a Hugging Face dataset with new clinical metadata.

    Args:
        repo_id (str): The ID of the repository on the Hugging Face Hub.
        metadata_csv_path (str): The path to the local metadata CSV file.
        base_config_name (str): The name of the base configuration to load.
        new_config_name (str): The name for the new configuration to be pushed.
        token (str): The Hugging Face API token for authentication.
    """
    print(f"Loading base dataset '{repo_id}' with configuration '{base_config_name}'...")
    # Load the base dataset configuration, ensuring it's not in streaming mode
    ds = load_dataset(repo_id, name=base_config_name)

    # If the loaded dataset is a single Dataset, wrap it in a DatasetDict.
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    print(f"Reading local metadata from '{metadata_csv_path}'...")
    # Read the local metadata file
    local_metadata_df = pd.read_csv(metadata_csv_path)

    # Use the first split to access features, assuming all splits have the same features.
    first_split_key = next(iter(ds.keys()))
    existing_features = ds[first_split_key].features
    
    # Get the unique values for categorical features from the dataframe and sort them
    marital_status_names = sorted(local_metadata_df['marital_status'].dropna().unique().tolist())
    race_names = sorted(local_metadata_df['race'].dropna().unique().tolist())
    eating_habits_names = sorted(local_metadata_df['eating_habits'].dropna().unique().tolist())
    mammography_names = sorted(local_metadata_df['mammography'].dropna().unique().tolist())
    radiotherapy_names = sorted(local_metadata_df['radiotherapy'].dropna().unique().tolist())
    plastic_surgery_names = sorted(local_metadata_df['plastic_surgery'].dropna().unique().tolist())
    prosthesis_names = sorted(local_metadata_df['prosthesis'].dropna().unique().tolist())
    biopsy_names = sorted(local_metadata_df['biopsy'].dropna().unique().tolist())
    use_of_hormone_replacement_names = sorted(local_metadata_df['use_of_hormone_replacement'].dropna().unique().tolist())
    nipple_changes_names = sorted(local_metadata_df['nipple_changes'].dropna().unique().tolist())
    is_there_signal_of_wart_on_breast_names = sorted(local_metadata_df['is_there_signal_of_wart_on_breast'].dropna().unique().tolist())
    protocol_smoked_names = sorted(local_metadata_df['protocol_smoked'].dropna().unique().tolist())
    protocol_drank_coffee_names = sorted(local_metadata_df['protocol_drank_coffee'].dropna().unique().tolist())
    protocol_consumed_alcohol_names = sorted(local_metadata_df['protocol_consumed_alcohol'].dropna().unique().tolist())
    protocol_physical_exercise_names = sorted(local_metadata_df['protocol_physical_exercise'].dropna().unique().tolist())
    protocol_put_some_pomade_deodorant_or_products_at_breasts_or_armpits_region_names = sorted(local_metadata_df['protocol_put_some_pomade_deodorant_or_products_at_breasts_or_armpits_region'].dropna().unique().tolist())

    # Define the full features for the new dataset
    new_features = Features({
        **existing_features,
        'record': Value('string'),
        'role': Value('string'),
        'age_current': Value('int64'),
        'age_at_visit': Value('int64'),
        'registration_date': Value('string'),
        'marital_status': ClassLabel(names=marital_status_names),
        'race': ClassLabel(names=race_names),
        'visit_date': Value('string'),
        'complaints': Value('string'),
        'symptoms': Value('string'),
        'signs': Value('string'),
        'last_menstrual_period': Value('string'),
        'menopause': Value('string'),
        'menarche': Value('int64'),
        'eating_habits': ClassLabel(names=eating_habits_names),
        'cancer_family': Value('string'),
        'family_history': Value('string'),
        'further_informations': Value('string'),
        'mammography': ClassLabel(names=mammography_names),
        'radiotherapy': ClassLabel(names=radiotherapy_names),
        'plastic_surgery': ClassLabel(names=plastic_surgery_names),
        'prosthesis': ClassLabel(names=prosthesis_names),
        'biopsy': ClassLabel(names=biopsy_names),
        'use_of_hormone_replacement': ClassLabel(names=use_of_hormone_replacement_names),
        'nipple_changes': ClassLabel(names=nipple_changes_names),
        'is_there_signal_of_wart_on_breast': ClassLabel(names=is_there_signal_of_wart_on_breast_names),
        'medical_further_informations': Value('string'),
        'body_temperature': Value('float64'),
        'protocol_smoked': ClassLabel(names=protocol_smoked_names),
        'protocol_drank_coffee': ClassLabel(names=protocol_drank_coffee_names),
        'protocol_consumed_alcohol': ClassLabel(names=protocol_consumed_alcohol_names),
        'protocol_physical_exercise': ClassLabel(names=protocol_physical_exercise_names),
        'protocol_put_some_pomade_deodorant_or_products_at_breasts_or_armpits_region': ClassLabel(names=protocol_put_some_pomade_deodorant_or_products_at_breasts_or_armpits_region_names),
    })

    print("Creating new DatasetDict with updated features...")
    new_ds = DatasetDict()
    
    # Select only the new clinical columns from the local metadata to avoid conflicts
    # and ensure patient_id is the key for merging.
    clinical_columns = [
        'patient_id', 'record', 'role', 'age_current', 'age_at_visit', 
        'registration_date', 'marital_status', 'race', 'visit_date', 
        'complaints', 'symptoms', 'signs', 'last_menstrual_period', 
        'menopause', 'menarche', 'eating_habits', 'cancer_family', 
        'family_history', 'further_informations', 'mammography', 
        'radiotherapy', 'plastic_surgery', 'prosthesis', 'biopsy', 
        'use_of_hormone_replacement', 'nipple_changes', 
        'is_there_signal_of_wart_on_breast', 'medical_further_informations', 
        'body_temperature', 'protocol_smoked', 'protocol_drank_coffee', 
        'protocol_consumed_alcohol', 'protocol_physical_exercise', 
        'protocol_put_some_pomade_deodorant_or_products_at_breasts_or_armpits_region'
    ]
    # Drop duplicates to have one row of clinical data per patient
    clinical_data_df = local_metadata_df[clinical_columns].drop_duplicates(subset=['patient_id']).reset_index(drop=True)

    # Create a lookup dictionary from the clinical data DataFrame for fast access
    clinical_data_df['patient_id'] = clinical_data_df['patient_id'].astype(str)
    clinical_data_lookup = clinical_data_df.set_index('patient_id').to_dict('index')

    def add_clinical_data(examples):
        """
        Applies clinical data to a batch of examples using a lookup table.
        """
        # Initialize lists for all new columns
        new_data = {col: [] for col in clinical_columns if col != 'patient_id'}
        
        # Get patient_ids from the batch and ensure they are strings
        patient_ids = [str(pid) for pid in examples['patient_id']]

        for pid in patient_ids:
            # Get the clinical data for the patient, or a default empty dict if not found
            patient_data = clinical_data_lookup.get(pid, {})
            
            # Append data for each clinical column, ensuring consistent string types for the map operation
            for col_name in new_data.keys():
                value = patient_data.get(col_name)
                # Convert potential NaN/None values to an empty string to prevent pyarrow type errors
                if pd.isna(value):
                    new_data[col_name].append("")
                else:
                    new_data[col_name].append(str(value))
        
        return new_data

    print("Applying clinical data to the dataset using .map() for memory efficiency...")
    # Use .map() to add the new columns in a memory-efficient way
    # remove_columns is used to avoid issues with pre-existing columns from the base config
    existing_cols_to_remove = [col for col in clinical_columns if col in ds[first_split_key].column_names and col != 'patient_id']
    
    updated_ds = ds.map(
        add_clinical_data, 
        batched=True, 
        remove_columns=existing_cols_to_remove
    )

    # Define the specific columns that need cleaning to avoid touching other data types.
    numeric_cols_to_clean = [
        col for col, feature in new_features.items() 
        if isinstance(feature, Value) and feature.dtype in ['int64', 'float64'] and col in clinical_columns
    ]
    categorical_cols_to_clean = [
        col for col, feature in new_features.items()
        if isinstance(feature, ClassLabel) and col in clinical_columns
    ]

    def clean_and_prepare_data(examples):
        """
        Cleans numeric and categorical columns before the final cast.
        - Converts float-like strings to integers for numeric columns.
        - Replaces empty strings in categorical columns with None for proper casting.
        """
        # Clean numeric columns
        for col in numeric_cols_to_clean:
            if col in examples:
                series = pd.Series(examples[col])
                numeric_series = pd.to_numeric(series, errors='coerce')
                
                feature_type = new_features[col].dtype
                if feature_type == 'int64':
                    examples[col] = numeric_series.fillna(0).astype('int64')
                elif feature_type == 'float64':
                    examples[col] = numeric_series.fillna(0.0).astype('float64')
        
        # Clean categorical columns
        for col in categorical_cols_to_clean:
            if col in examples:
                # Replace empty strings with None so .cast() can handle them as missing values
                examples[col] = [val if val != "" else None for val in examples[col]]

        return examples

    print("Cleaning and preparing data before final cast...")
    cleaned_ds = updated_ds.map(clean_and_prepare_data, batched=True)


    print("Casting features to the correct data types...")
    # Cast the entire dataset to the new features schema
    new_ds = cleaned_ds.cast(new_features)


    print(f"Pushing new configuration '{new_config_name}' to '{repo_id}'...")
    new_ds.push_to_hub(
        repo_id=repo_id,
        config_name=new_config_name,
        commit_message=f"Add '{new_config_name}' configuration with full clinical metadata"
    )
    print("Successfully updated dataset on the Hugging Face Hub.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update a Hugging Face dataset with clinical data.")
    parser.add_argument("--repo_id", type=str, default="SemilleroCV/DMR-IR", help="The ID of the repository on the Hugging Face Hub.")
    parser.add_argument("--metadata_csv", type=str, default="data/DMR-IR/dmr_ir_complete_metadata.csv", help="Path to the local metadata CSV file.")
    parser.add_argument("--base_config_name", type=str, default="with_full_metadata", help="The name of the base configuration to load.")
    parser.add_argument("--new_config_name", type=str, default="with_full_clinical_metadata", help="The name for the new configuration.")

    args = parser.parse_args()


    update_dataset_with_clinical_data(
        repo_id=args.repo_id,
        metadata_csv_path=args.metadata_csv,
        base_config_name=args.base_config_name,
        new_config_name=args.new_config_name,
    )
