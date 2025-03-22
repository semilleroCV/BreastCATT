import json

def convert_json_to_sentence(data):
    # Extract data from each section of the JSON
    demographic = data.get("demographic", {})
    personal_history = data.get("personal_history", {})
    clinical_history = data.get("clinical_history", {})
    protocol = data.get("protocol", {})

    # Demographic data
    age = demographic.get("age", "unknown")
    marital_status = demographic.get("marital_status", "")
    race = demographic.get("race", "")
    registration_date = demographic.get("registration_date", "")
    
    # If the race is empty or "more", consider it as unspecified
    if not race or race.lower() == "more":
        race_text = "of unspecified race"
    else:
        race_text = f"of {race} race"
    
    # Personal history
    eating_habits = personal_history.get("eating habits", "")
    last_menstrual_period = personal_history.get("last menstrual period", "")
    menarche = personal_history.get("menarche", "")
    signs = personal_history.get("signs", "")
    symptoms = personal_history.get("symptoms", "")
    
    # Clinical history
    mammography = clinical_history.get("mammography", "")
    biopsy = clinical_history.get("biopsy", "")
    plastic_surgery = clinical_history.get("plastic surgery", "")
    prosthesis = clinical_history.get("prosthesis", "")
    radiotherapy = clinical_history.get("radiotherapy", "")
    hormone_replacement = clinical_history.get("use of hormone replacement", "")
    
    # Protocol data
    body_temperature = protocol.get("body temperature", "")
    consumed_alcohol = protocol.get("consumed alcohol", "")
    drank_coffee = protocol.get("drank coffee", "")
    physical_exercise = protocol.get("physical exercise", "")
    smoked = protocol.get("smoked", "")
    
    # Build the sentence step by step
    sentence_parts = []
    
    # Demographic information
    sentence_parts.append(f"Patient is {age} years old")
    if marital_status:
        sentence_parts.append(f", {marital_status}")
    sentence_parts.append(f" {race_text}")
    if registration_date:
        sentence_parts.append(f", registered on {registration_date}")
    sentence_parts.append(".")
    
    # Personal history
    if eating_habits:
        # For this example, if 'fat' is provided, we consider it as a high-fat diet.
        if eating_habits.lower() == "fat":
            sentence_parts.append(" Has a high-fat diet.")
        else:
            sentence_parts.append(f" Has eating habits rich in {eating_habits}.")
    if last_menstrual_period and menarche:
        sentence_parts.append(f" Their last menstruation was {last_menstrual_period} and menarche occurred {menarche}.")
    elif last_menstrual_period:
        sentence_parts.append(f" Their last menstruation was {last_menstrual_period}.")
    elif menarche:
        sentence_parts.append(f" Menarche occurred {menarche}.")
    
    # Indicate missing signs and symptoms if applicable
    if not signs and not symptoms:
        sentence_parts.append(" No relevant clinical signs or symptoms were reported.")
    
    # Clinical history: list items with no information
    missing_clinical = []
    if not mammography:
        missing_clinical.append("mammography")
    if not biopsy:
        missing_clinical.append("biopsy")
    if not plastic_surgery:
        missing_clinical.append("plastic surgery")
    if not prosthesis:
        missing_clinical.append("prosthesis")
    if not radiotherapy:
        missing_clinical.append("radiotherapy")
    if not hormone_replacement:
        missing_clinical.append("use of hormone replacement")
    if missing_clinical:
        sentence_parts.append(" No information is provided for " + ", ".join(missing_clinical) + ".")
    
    # Protocol information
    if body_temperature:
        sentence_parts.append(f" The recorded body temperature is {body_temperature}°C.")
    
    missing_protocol = []
    if not consumed_alcohol:
        missing_protocol.append("alcohol consumption")
    if not drank_coffee:
        missing_protocol.append("coffee drinking")
    if not smoked:
        missing_protocol.append("smoking")
    if not physical_exercise:
        missing_protocol.append("recent physical exercise")
    if missing_protocol:
        sentence_parts.append(" No mention is made of " + ", ".join(missing_protocol) + ".")
    
    return "".join(sentence_parts)

def generate_category_prompts(data):
    # Extract sections
    demographic = data.get("demographic", {})
    personal_history = data.get("personal_history", {})
    clinical_history = data.get("clinical_history", {})
    protocol = data.get("protocol", {})
    
    # Risk factors prompt
    risk_factors_parts = []
    risk_factors_parts.append(f"Age: {demographic.get('age', 'unknown')}")
    risk_factors_parts.append(f"Eating habits: {personal_history.get('eating habits', 'not specified')}")
    risk_factors_parts.append(f"Family history of cancer: {personal_history.get('family history', 'not specified')}")
    risk_factors_parts.append(f"Radiotherapy treatment: {clinical_history.get('radiotherapy', 'not specified')}")
    risk_factors_parts.append(f"Hormone replacement: {clinical_history.get('use of hormone replacement', 'not specified')}")
    risk_factors_parts.append(f"Age at menarche: {personal_history.get('menarche', 'not specified')}")
    risk_factors_parts.append(f"Age at menopause: {personal_history.get('menopause', 'not specified')}")
    risk_factors_parts.append(f"Diabetes history: {personal_history.get('diabetes', 'not specified')}")
    risk_factors_parts.append(f"Presence of nodules: {personal_history.get('nodules', 'not specified')}")
    risk_factors_prompt = "Risk factors: " + "; ".join(risk_factors_parts) + "."
    
    # Complementary features prompt
    comp_parts = []
    comp_parts.append(f"Prosthesis: {clinical_history.get('prosthesis', 'not specified')}")
    comp_parts.append(f"Signals of wart on breasts: {clinical_history.get('is there signal of wart on breast', 'not specified')}")
    comp_parts.append(f"Nipple changes: {clinical_history.get('nipple changes', 'not specified')}")
    comp_parts.append(f"Mastectomy: {clinical_history.get('mastectomy', 'not specified')}")
    comp_prompt = "Complementary features: " + "; ".join(comp_parts) + "."
    
    # Protocol features prompt
    prot_parts = []
    prot_parts.append(f"Smoked: {protocol.get('smoked', 'not specified')}")
    prot_parts.append(f"Drank coffee: {protocol.get('drank coffee', 'not specified')}")
    prot_parts.append(f"Consumed alcohol: {protocol.get('consumed alcohol', 'not specified')}")
    prot_parts.append(f"Physical exercise: {protocol.get('physical exercise', 'not specified')}")
    prot_parts.append(f"Use of deodorant/products: {protocol.get('deodorant or products at breasts or armpits region', 'not specified')}")
    prot_prompt = "Protocol features: " + "; ".join(prot_parts) + "."
    
    return {
        "risk_factors": risk_factors_prompt,
        "complementary_features": comp_prompt,
        "protocol_features": prot_prompt
    }