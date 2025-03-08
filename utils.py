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