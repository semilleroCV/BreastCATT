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