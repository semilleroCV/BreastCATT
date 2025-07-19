import argparse
import pandas as pd
from tqdm import tqdm
import re

def create_prompt_from_csv_row(row, include_demographic=True, include_personal_history=True, include_medical_history=True, include_protocol=True):
    """
    Generates a natural language prompt from a row of the clinical data CSV,
    with options to include different sections of information.
    """
    sentence_parts = []

    # --- Demographic Information ---
    if include_demographic:
        age = row.get("age_at_visit", "unknown")
        race = row.get("race", "")
        if not race or pd.isna(race):
            race_text = "of unspecified race"
        else:
            race_text = f"and identifies as {race.lower()}"
        sentence_parts.append(f"Patient is {age} years old {race_text}.")

    # --- Personal and Clinical History ---
    if include_personal_history:
        menarche = row.get("menarche")
        lmp = row.get("last_menstrual_period")
        menopause_info = row.get("menopause")
        symptoms = row.get("symptoms")
        signs = row.get("signs")
        complaints = row.get("complaints")
        family_history = row.get("family_history")
        cancer_family = row.get("cancer_family")
        eating_habits = row.get("eating_habits")

        if menarche and pd.notna(menarche):
            try:
                sentence_parts.append(f"Menarche occurred at age {int(float(menarche))}.")
            except (ValueError, TypeError):
                sentence_parts.append(f"Menarche information: {menarche}.")

        if lmp and pd.notna(lmp):
            try:
                # Try to interpret as an age first
                sentence_parts.append(f"Her last menstrual period was at age {int(float(lmp))}.")
            except (ValueError, TypeError):
                # Otherwise, treat as a date or other text
                if str(lmp).lower() == 'had a hysterectomy':
                    sentence_parts.append(f"Had a hysterectomy.")
                # else:
                #     sentence_parts.append(f"Last menstrual period: {lmp}.")

        if menopause_info and pd.notna(menopause_info):
            try:
                # Handles ages like 50, 50.0, etc.
                menopause_age = int(float(menopause_info))
                sentence_parts.append(f"Entered menopause at age {menopause_age}.")
            except (ValueError, TypeError):
                # Handle string values like 'Yes', 'No', etc.
                menopause_str = str(menopause_info).strip().lower()
                if menopause_str == 'yes':
                    sentence_parts.append("Patient is menopausal.")
                elif menopause_str == 'no':
                    sentence_parts.append("Patient is not menopausal.")
                elif menopause_str != 'no': # Only add if it's not 'No' and not numeric
                    sentence_parts.append(f"Menopause information: {menopause_info}.")

        if complaints and pd.notna(complaints):
            if complaints not in ['-', 'No']:
                sentence_parts.append(f"Patient complaints: {complaints.lower()}.")
            elif complaints == 'No':
                sentence_parts.append("No complaints were reported.")

        if symptoms and pd.notna(symptoms):
            if symptoms not in ['-', 'No']:
                sentence_parts.append(f"Reported symptoms include: {symptoms.lower()}.")
            elif symptoms == 'No':
                sentence_parts.append("No symptoms were reported.")
            
        if signs and pd.notna(signs):
            if signs not in ['-', 'No']:
                sentence_parts.append(f"Observed signs include: {signs.lower()}.")
            elif signs == 'No':
                sentence_parts.append("No signs were observed.")

        if family_history and pd.notna(family_history) and family_history not in ['-', 'No']:
                sentence_parts.append(f"The patient has a family history of {family_history.lower()}.")
        
        if cancer_family and pd.notna(cancer_family) and cancer_family not in ['-', 'No']:
                sentence_parts.append(f"Family history of cancer includes: {cancer_family.lower()}.")

        if eating_habits and pd.notna(eating_habits) and eating_habits not in ['-', 'No']:
            if eating_habits == "No fat":
                sentence_parts.append(f"Has eating habits fat-free.")
            else:
              sentence_parts.append(f"Has eating habits {eating_habits.lower()}.")

    # --- Medical History / Procedures ---
    if include_medical_history:
        procedure_map = {
            "mammography": "mammography",
            "radiotherapy": "radiotherapy",
            "plastic_surgery": "plastic surgery",
            "prosthesis": "prosthesis",
            "biopsy": "biopsy",
            "use_of_hormone_replacement": "hormone replacement therapy",
            "nipple_changes": "nipple changes",
            "is_there_signal_of_wart_on_breast": "signal of wart on breast"
        }
        
        procedures = []
        for col, name in procedure_map.items():
            formatted_procedure = _format_procedure(name, row.get(col))
            if formatted_procedure:
                procedures.append(formatted_procedure)

        if procedures:
            sentence_parts.append(f"Has a medical history of: {'; '.join(procedures)}.")
        else:
            sentence_parts.append("Has no medical history of procedures.")

    # --- Protocol Information ---
    if include_protocol:
        protocol_parts = []
        body_temp = row.get("body_temperature")
        smoked = row.get("protocol_smoked")
        coffee = row.get("protocol_drank_coffee")
        alcohol = row.get("protocol_consumed_alcohol")
        exercise = row.get("protocol_physical_exercise")
        products = row.get("protocol_put_some_pomade_deodorant_or_products_at_breasts_or_armpits_region")

        if body_temp and pd.notna(body_temp):
            body_temp_text = f"body temperature was {body_temp} degrees Celsius"
        
        if smoked and pd.notna(smoked) and smoked == 'Yes':
            protocol_parts.append("smoked")
        
        if coffee and pd.notna(coffee) and coffee == 'Yes':
            protocol_parts.append("drank coffee")

        if alcohol and pd.notna(alcohol) and alcohol == 'Yes':
            protocol_parts.append("consumed alcohol")
            
        if exercise and pd.notna(exercise) and exercise == 'Yes':
            protocol_parts.append("performed physical exercise")
            
        if products and pd.notna(products) and products == 'Yes':
            protocol_parts.append("used products on breasts or armpits")

        if not protocol_parts and body_temp and pd.notna(body_temp):
            sentence_parts.append(f"Regarding the exam protocol, {body_temp_text}.")
        elif protocol_parts and body_temp and pd.notna(body_temp):
            sentence_parts.append(f"Regarding the exam protocol, {body_temp_text} and the patient {', '.join(protocol_parts)} prior to exam.")
        else:
            sentence_parts.append("No protocol information was provided.")

    return " ".join(sentence_parts)

def _format_procedure(procedure_name, value):
    """Helper function to format a single medical procedure sentence part."""
    if not value or pd.isna(value):
        return None

    val_lower = str(value).lower().strip()

    if val_lower == 'no':
        return None

    if val_lower == 'yes':
        return procedure_name

    # Handle cases like "Yes, right breast", "yes, both", "Biopsy, yes, left breast"
    # Use regex to capture optional procedure name and location
    match = re.match(r'(?:[\w\s]+,\s*)?yes,\s*(right|left|both)', val_lower)
    if match:
        location = match.group(1)
        
        # Special handling for 'signal of wart on breast' to avoid "on breast on the..."
        if 'signal of wart' in procedure_name:
            procedure_name = 'wart-like lesion'
        elif 'biopsy' in procedure_name:
            if location in ['right', 'left']:
                return f"{procedure_name} of the {location} breast"
            elif location == 'both':
                return f"{procedure_name} of both breasts"

        if location in ['right', 'left']:
            return f"{procedure_name} on the {location} breast"
        elif location == 'both':
            return f"{procedure_name} on both breasts"

    # Fallback for original logic and other cases
    if 'yes, ' in val_lower:
        location = val_lower.replace('yes, ', '').replace(' breast', '').replace(' breasts', '')

        if 'signal of wart' in procedure_name and location:
            procedure_name = procedure_name.replace(' on breast', '')
            
        if location in ['right', 'left']:
            return f"{procedure_name} on the {location} breast"
        elif location == 'both':
            return f"{procedure_name} on both breasts"
    
    # Fallback for other non-No/Yes values that don't match the pattern
    return f"{procedure_name}, {val_lower}"

def generate_prompts_and_embeddings(csv_path, output_path, prompts_only=False, 
                                     include_demographic=True,
                                     include_personal_history=True,
                                     include_medical_history=True, include_protocol=True):
    """
    Reads a CSV, generates prompts, optionally creates embeddings, and saves the output.
    """
    # 1. Read CSV
    print(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    all_prompts = []
    all_labels = []

    # 2. Generate prompts
    print("Generating prompts...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        prompt = create_prompt_from_csv_row(row, include_demographic, include_personal_history, include_medical_history, include_protocol)
        all_prompts.append(prompt)
        all_labels.append(row.get('diagnosis', -1))

    if prompts_only:
        # Save prompts, labels, and patient_ids to a CSV file
        output_df = pd.DataFrame({
            'patient_id': df['patient_id'].tolist(),
            'label': all_labels,
            'text': all_prompts
        })
        
        # Ensure output path has .csv extension
        if not output_path.endswith('.csv'):
            output_path = output_path.rsplit('.', 1)[0] + '.csv'
            print(f"Output path adjusted to {output_path} for CSV format.")

        output_df.to_csv(output_path, index=False)
        print(f"Prompts saved to {output_path}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text prompts and/or embeddings from clinical data.")
    parser.add_argument("--csv_path", type=str, default="data/clinical_data.csv", help="Path to the merged clinical data CSV.")
    parser.add_argument("--output_path", type=str, default="data/prompt_embeddings.pt", help="Path to save the output file. Will be .csv for prompts_only, .pt otherwise.")
    
    # Flags for controlling content
    parser.add_argument("--prompts_only", action='store_true', help="Only generate and save prompts, skip embedding generation.")
    parser.add_argument('--no_demographic', action='store_false', dest='include_demographic', help='Do not include demographic information in the prompt.')
    parser.add_argument('--no_personal_history', action='store_false', dest='include_personal_history', help='Do not include personal history in the prompt.')
    parser.add_argument('--no_medical_history', action='store_false', dest='include_medical_history', help='Do not include medical history and procedures in the prompt.')
    parser.add_argument('--no_protocol', action='store_false', dest='include_protocol', help='Do not include protocol information in the prompt.')
    
    args = parser.parse_args()
    
    generate_prompts_and_embeddings(
        args.csv_path, 
        args.output_path, 
        prompts_only=args.prompts_only,
        include_demographic=args.include_demographic,
        include_personal_history=args.include_personal_history,
        include_medical_history=args.include_medical_history,
        include_protocol=args.include_protocol
    )
