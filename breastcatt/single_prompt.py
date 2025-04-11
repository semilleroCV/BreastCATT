def convert_json_to_sentence(data, include_demographic=True, include_personal_history=True, include_medical_history=True, include_protocol=True):
    # Extract data from each section of the JSON
    demographic = data.get("demographic", {})
    personal_history = data.get("personal_history", {})
    clinical_history = data.get("clinical_history", {})
    protocol = data.get("protocol", {})

    # Demographic data
    age = demographic.get("age", "unknown")
    race = demographic.get("race", "")
    
    if not race:
        race_text = "of unspecified race"
    elif race.lower() == "more":
        race_text = "of mestizo race"
    elif race.lower() == "amarela":
        race_text = "of asian race"
    else:
        race_text = f"of {race} race"
    
    # Personal history
    eating_habits = personal_history.get("eating habits", "")
    menarche = personal_history.get("menarche", "")
    signs = personal_history.get("signs", "")
    symptoms = personal_history.get("symptoms", "")
    complaints = personal_history.get("complaints", "")
    family_history = personal_history.get("family history", "")
    
    # Clinical (medical) history
    radiotherapy = clinical_history.get("radiotherapy", "")
    hormone_replacement = clinical_history.get("use of hormone replacement", "")
    nipple_changes = clinical_history.get("nipple changes", "")
    
    # Protocol data
    body_temperature = protocol.get("body temperature", "")
    consumed_alcohol = protocol.get("consumed alcohol", "")
    drank_coffee = protocol.get("drank coffee", "")
    physical_exercise = protocol.get("physical exercise", "")
    smoked = protocol.get("smoked", "")
    
    # Build the sentence step by step
    sentence_parts = []
    
    if include_demographic:
        sentence_parts.append(f"Patient is {age} years old")
        sentence_parts.append(f" {race_text}")
        sentence_parts.append(".")
    
    if include_personal_history:
        if eating_habits:
            if eating_habits.lower() == "fat":
                sentence_parts.append(" Has a high-fat diet.")
            else:
                sentence_parts.append(f" Has eating habits {eating_habits}.")
        elif menarche:
            sentence_parts.append(f" Menarche occurred {menarche}.")
        if complaints:
            sentence_parts.append(f" Complaints reported were {complaints}.")
        if family_history:
            sentence_parts.append(f" Family history reported {family_history}.")
        if signs or symptoms:
            s_and_s = []
            if signs:
                s_and_s.append(f"{signs}")
            if symptoms:
                s_and_s.append(f"{symptoms}")
            sentence_parts.append(" Additionally, patient reported " + " and ".join(s_and_s) + ".")
        if not signs and not symptoms:
            sentence_parts.append(" No relevant clinical symptoms were reported.")
    
    if include_medical_history:
        missing_clinical = []
        if not radiotherapy:
            missing_clinical.append("radiotherapy")
        if not hormone_replacement:
            missing_clinical.append("use of hormone replacement")
        if missing_clinical:
            sentence_parts.append(" No information provided for " + ", ".join(missing_clinical) + ".")
        provided_phrases = []
        if radiotherapy:
            provided_phrases.append(f"radiotherapy was {radiotherapy}")
        if hormone_replacement:
            provided_phrases.append(f"use of hormone replacement was {hormone_replacement}")
        if nipple_changes:
            provided_phrases.append(f"nipple changes were {nipple_changes}")
        if provided_phrases:
            narrative = " In the clinical history, " + ", ".join(provided_phrases) + "."
            sentence_parts.append(narrative)
    
    if include_protocol:
        protocol_phrases = []
        if body_temperature:
            protocol_phrases.append(f"a body temperature of {body_temperature} degrees Celsius")
        if consumed_alcohol:
            protocol_phrases.append(f"alcohol consumption was {consumed_alcohol}")
        if drank_coffee:
            protocol_phrases.append(f"coffee drinking was {drank_coffee}")
        if smoked:
            protocol_phrases.append(f"smoking was {smoked}")
        if physical_exercise:
            protocol_phrases.append(f"physical exercise was {physical_exercise}")
        if protocol_phrases:
            narrative = " Regarding the protocol, " + ", ".join(protocol_phrases) + "."
            sentence_parts.append(narrative)
        else:
            sentence_parts.append(" No protocol information provided.")
    
    return "".join(sentence_parts)