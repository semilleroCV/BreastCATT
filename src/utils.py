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
    # registration_date = demographic.get("registration_date", "")
    
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
    # if registration_date:
    #     sentence_parts.append(f", registered on {registration_date}")
    sentence_parts.append(".")
    
    # Personal history
    if eating_habits:
        # For this example, if 'fat' is provided, we consider it as a high-fat diet.
        if eating_habits.lower() == "fat":
            sentence_parts.append(" Has a high-fat diet.")
        else:
            sentence_parts.append(f" Has eating habits {eating_habits}.")
    if last_menstrual_period and menarche:
        sentence_parts.append(f" Last menstruation was {last_menstrual_period} and menarche occurred {menarche}.")
    elif last_menstrual_period:
        sentence_parts.append(f" Last menstruation was {last_menstrual_period}.")
    elif menarche:
        sentence_parts.append(f" Menarche occurred {menarche}.")
    
    # Indicate missing signs and symptoms if applicable (DA 85.71%, 1% más que sino se le pasa)
    if not signs and not symptoms:
        sentence_parts.append(" No relevant clinical symptoms were reported.")
    
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
    # if missing_clinical: # NO FUNCION PORQUE TIENE MUCHO TEXTO
    #     sentence_parts.append(" No information provided for " + ", ".join(missing_clinical) + ".")
    
    # Protocol information
    if body_temperature:
        sentence_parts.append(f" The body temperature is {body_temperature}°C.")
    
    missing_protocol = []
    if not consumed_alcohol:
        missing_protocol.append("alcohol consumption")
    if not drank_coffee:
        missing_protocol.append("coffee drinking")
    if not smoked:
        missing_protocol.append("smoking")
    if not physical_exercise:
        missing_protocol.append("recent physical exercise")
    if missing_protocol: # Agregando esto Y "No relevant clinical symptoms were reported." da más de 90% en algunos casos
        sentence_parts.append(" No mention of " + ", ".join(missing_protocol) + ".")
    
    return "".join(sentence_parts)

def generate_category_prompts(data):
    demographic = data.get("demographic", {})
    personal = data.get("personal_history", {})
    clinical = data.get("clinical_history", {})
    protocol = data.get("protocol", {})
    
    # Risk factors narrative
    age = demographic.get("age", "unknown")
    eating = personal.get("eating habits", "").strip()
    family = personal.get("family history", "").strip()
    radiotherapy = clinical.get("radiotherapy", "").strip()
    hormone = clinical.get("use of hormone replacement", "").strip()
    menarche = personal.get("menarche", "").strip()
    menopause = personal.get("menopause", "").strip()
    diabetes = personal.get("diabetes", "").strip()
    nodules = personal.get("nodules", "").strip()
    
    risk_sentences = [f"The patient is {age} years old."]
    if eating:
        risk_sentences.append(f"Her eating habits are {eating}.")
    if family:
        risk_sentences.append(f"There is a family history of cancer noted as {family}.")
    if radiotherapy:
        risk_sentences.append(f"Radiotherapy treatment is reported as {radiotherapy}.")
    if hormone:
        risk_sentences.append(f"Hormone replacement therapy is indicated as {hormone}.")
    if menarche:
        risk_sentences.append(f"Menarche occurred {menarche}.")
    if menopause:
        risk_sentences.append(f"Menopause occurred {menopause}.")
    if diabetes:
        risk_sentences.append(f"Diabetes history is reported as {diabetes}.")
    if nodules:
        risk_sentences.append(f"Nodules have been indicated as {nodules}.")
    risk_paragraph = " ".join(risk_sentences)
    
    # Complementary features narrative
    prosthesis = clinical.get("prosthesis", "").strip()
    wart = clinical.get("is there signal of wart on breast", "").strip()
    nipple = clinical.get("nipple changes", "").strip()
    
    comp_sentences = []
    if prosthesis:
        comp_sentences.append(f"Prosthesis status is {prosthesis}.")
    if wart:
        comp_sentences.append(f"Signals of warts on the breasts are described as {wart}.")
    if nipple:
        comp_sentences.append(f"Nipple changes are observed as {nipple}.")
    if not comp_sentences:
        comp_sentences.append("No complementary features were reported.")
    comp_paragraph = " ".join(comp_sentences)
    
    # Protocol features narrative
    smoked = protocol.get("smoked", "").strip()
    coffee = protocol.get("drank coffee", "").strip()
    alcohol = protocol.get("consumed alcohol", "").strip()
    exercise = protocol.get("physical exercise", "").strip()
    deodorant = protocol.get("deodorant or products at breasts or armpits region", "").strip()
    body_temperature = protocol.get("body temperature", "").strip()
    
    protocol_sentences = []
    if body_temperature:
        protocol_sentences.append(f"Body temperature is recorded as {body_temperature}°C.")
    if smoked:
        protocol_sentences.append(f"Smoking status is reported as {smoked}.")
    if coffee:
        protocol_sentences.append(f"Coffee consumption is noted as {coffee}.")
    if alcohol:
        protocol_sentences.append(f"Alcohol consumption is indicated as {alcohol}.")
    if exercise:
        protocol_sentences.append(f"Physical exercise is mentioned as {exercise}.")
    if deodorant:
        protocol_sentences.append(f"Use of deodorant or related products is reported as {deodorant}.")
    if not protocol_sentences:
        protocol_sentences.append("No protocol features were reported.")
    protocol_paragraph = " ".join(protocol_sentences)
    
    return {
        "risk_factors": risk_paragraph,
        "complementary_features": comp_paragraph,
        "protocol_features": protocol_paragraph
    }