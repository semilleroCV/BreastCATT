import requests
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import pandas as pd
from tqdm import tqdm
import random
import time
from dateutil.parser import parse as parse_date, ParserError
import re
from datetime import datetime
from deep_translator import GoogleTranslator
import argparse

from tabulate import tabulate
import numpy as np

def translate_text(text):
    """Translates text to English, with special handling for common words."""
    if not text or not isinstance(text, str) or not re.search('[a-zA-Z]', text):
        return text
    
    # Fix encoding artifacts and normalize
    text_lower = text.lower().replace('?', 'ã').strip()
    text_lower = text_lower.replace('::', '').strip()

    # Yes/No variations
    if text_lower in ['sim', 'yes', 's', 'sim.']: return 'Yes'
    if text_lower in ['não', 'nao', 'no', 'n', 'não tem', 'não.']: return 'No'
    
    # Marital status
    if text_lower in ['viuvo', 'viúva', 'widowed', 'widow', 'viuva']: return 'Widow'
    if text_lower in ['solteiro', 'solteira', 'single']: return 'Single'
    if text_lower in ['casado', 'casada', 'married']: return 'Married'
    if text_lower in ['divorciado', 'divorciada', 'divorced']: return 'Divorced'

    # Race
    if text_lower in ['pardo', 'parda', 'brown']: return 'Multiracial'
    if text_lower in ['branca', 'white']: return 'White'
    if text_lower in ['negra', 'preta', 'black']: return 'Black'
    if text_lower in ['indigena']: return 'Indigenous'
    if text_lower in ['mulattos']: return 'Mulatto'
    if text_lower in ['amarela']: return 'Asian'

    # Eating habits
    if 'pobre em gordura' in text_lower or 'low in fat' in text_lower: return 'Low in fat'
    if 'rica em gordura' in text_lower or 'fatty diet' in text_lower: return 'High in fat'
    if 'sem gordura' in text_lower or 'no fat' in text_lower: return 'No fat'

    # Family
    if text_lower == 'mãe': return 'Mother'
    if text_lower == 'pai': return 'Father'
    if text_lower == 'irmã': return 'Sister'
    if text_lower == 'irmão': return 'Brother'
    if text_lower == 'tia': return 'Aunt'
    if text_lower == 'tio': return 'Uncle'
    if text_lower == 'avó': return 'Grandmother'
    if text_lower == 'avô': return 'Grandfather'
    if text_lower == 'prima': return 'Cousin (female)'
    if text_lower == 'primo': return 'Cousin (male)'
    if 'avo paterna (cancer de mama)' in text_lower: return 'Paternal grandmother (breast cancer)'
    
    # Last menstrual period
    if text_lower == 'fez exterectomia': return "had a hysterectomy"
    if text_lower == 'operou': return "surgery"

    # Symptoms
    if text_lower == 'dor': return 'Pain'
    if 'dores em periodo menstrual' in text_lower: return 'Pain during menstrual period'
    if 'sente dores na mama esquerda' in text_lower: return 'Feels pain in the left breast'
    if 'sente dores na mama direita' in text_lower: return 'Feels pain in the right breast'
    if 'sente dores na mama' in text_lower: return 'Feels pain in the breast'

    # Complaints
    if 'dores e quando presionava os seios saia uma massa' in text_lower: return 'Pain and when pressing on the breasts, a mass came out'

    try:
        # For longer texts that need full translation
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text


def _translate_month_to_english(date_str):
    """Helper to translate month names in a date string to English."""
    month_map = {
        'enero': 'january', 'janeiro': 'january',
        'febrero': 'february', 'fevereiro': 'february',
        'marzo': 'march', 'março': 'march',
        'abril': 'april',
        'mayo': 'may', 'maio': 'may',
        'junio': 'june', 'junho': 'june',
        'julio': 'july', 'julho': 'july',
        'agosto': 'august',
        'septiembre': 'september', 'setembro': 'september',
        'octubre': 'october', 'outubro': 'october', 'outubbro': 'october', # Added typo
        'noviembre': 'november', 'novembro': 'november',
        'diciembre': 'december', 'dezembro': 'december'
    }
    # Normalize by removing ' de ' which is common in Spanish/Portuguese
    date_str_lower = date_str.lower().replace(' de ', ' ')
    for esp, eng in month_map.items():
        if esp in date_str_lower:
            return date_str_lower.replace(esp, eng)
    return date_str_lower


def has_value(d, key):
    return key in d and d[key] is not None and not (isinstance(d[key], float) and np.isnan(d[key]))


def parse_last_menstrual_period(value_str, patient_data=None):
    """
    Parses the 'Last Menstrual Period' field, expecting a date.
    Handles partial dates by inferring the year from the visit date.
    """
    if not value_str or value_str.strip() in ['-', '—', '--'] or not isinstance(value_str, str):
        return None

    value_str_lower = value_str.lower().strip()

    # --- Priority 1: Handle "Faz X anos" (It's been X years) ---
    if 'faz' in value_str_lower and ('ano' in value_str_lower or 'year' in value_str_lower):
        years_ago_match = re.search(r'(\d+)', value_str_lower)
        if years_ago_match and patient_data and 'age_at_visit' in patient_data and has_value(patient_data, 'age_at_visit'):
            try:
                years_ago = int(years_ago_match.group(1))
                age_at_visit = int(patient_data['age_at_visit'])
                # This calculates the age at which the event (LMP) occurred.
                return int(age_at_visit - years_ago)
            except (ValueError, TypeError):
                pass # Fall through if data is missing or not numeric

    # --- Priority 2: Check for explicit age markers like "anos" or "years" ---
    if "ano" in value_str_lower or "year" in value_str_lower or "naos" in value_str_lower:
        age_match = re.search(r'(\d+)', value_str_lower)
        if age_match:
            age = int(age_match.group(1))
            if age < 100: # Sanity check
                return age
            
    # --- Priority 3: Check for a direct age (e.g., "43", "52") ---
    age_match = re.search(r'^(\d+)$', value_str_lower)
    if age_match:
        age = int(age_match.group(1))
        # Avoid capturing years, but allow ages.
        if age < 100 and not (1900 < age < 2100):
             return age
    
    # --- Priority 4: A value with separators or month names is very likely a date. ---
    is_likely_date = re.search(r'[a-zA-Z]', value_str_lower) or re.search(r'[/.-]', value_str_lower)
    if is_likely_date:
        try:
            parseable_date_str = _translate_month_to_english(value_str)
            default_date = None
            # If no year is present, use the visit date to infer it.
            if patient_data and 'visit_date' in patient_data and has_value(patient_data, 'visit_date'):
                try:
                    # Set the visit date as the default for dateutil to use if parts are missing
                    default_date = datetime.strptime(patient_data['visit_date'], '%Y-%m-%d')
                except (ValueError, TypeError):
                    pass
            
            # The dayfirst=True argument helps resolve ambiguity for formats like "DD/MM"
            dt = parse_date(parseable_date_str, fuzzy=True, default=default_date, dayfirst=True)

            # If we still couldn't determine a year and it defaulted to the current year, it's ambiguous.
            if not default_date and dt.year == datetime.now().year and not re.search(r'\d{4}', value_str):
                 return translate_text(value_str)

            return dt.strftime('%Y-%m-%d')
        except (ParserError, TypeError, ValueError):
            # If parsing fails, return the original translated text.
            return translate_text(value_str)
    
    # --- Priority 5: Date-based AGE calculation (if value is a year) ---
    if patient_data and 'age_at_visit' in patient_data and 'visit_date' in patient_data:
        try:
            age_at_visit = int(patient_data['age_at_visit'])
            visit_date = datetime.strptime(patient_data['visit_date'], '%Y-%m-%d')
            birth_year = visit_date.year - age_at_visit
            
            menopause_year = int(value_str.strip())
            if 1900 < menopause_year < 2100:
                menopause_age = menopause_year - birth_year
                return int(menopause_age)
        except (ValueError, TypeError):
             pass

    # If it's not a date, it might be other text.
    return translate_text(value_str)


def parse_menopause(value_str, patient_data=None):
    """
    Parses the 'Menopause' field, expecting an age, a status, or descriptive text.
    """
    if not value_str or value_str.strip() in ['-', '—', '--'] or not isinstance(value_str, str):
        return None

    value_str_lower = value_str.lower().strip()

    # --- Priority 1: Check for explicit age markers like "anos" or "years" ---
    if "ano" in value_str_lower or "year" in value_str_lower or "naos" in value_str_lower:
        age_match = re.search(r'(\d+)', value_str_lower)
        if age_match:
            age = int(age_match.group(1))
            if age < 100: # Sanity check
                return age

    # --- Priority 2: Check for a direct age (e.g., "43", "52") ---
    age_match = re.search(r'^(\d+)$', value_str_lower)
    if age_match:
        age = int(age_match.group(1))
        # Avoid capturing years, but allow ages.
        if age < 100 and not (1900 < age < 2100):
             return age

    # --- Priority 3: Check for simple "Yes/No" status ---
    if value_str_lower in ['sim', 'yes']:
        return 'Yes'
    if value_str_lower in ['não', 'nao', 'no']:
        return 'No'

    # --- Priority 4: Calculation logic based on patient's age at visit ---
    if patient_data and 'age_at_visit' in patient_data:
        try:
            age_at_visit = int(patient_data['age_at_visit'])
            # Handle "+ de X anos" (more than X years ago)
            if '+ de' in value_str_lower and 'anos' in value_str_lower:
                years_ago_match = re.search(r'\d+', value_str_lower)
                if years_ago_match:
                    years_ago = int(years_ago_match.group(0))
                    return int(age_at_visit - years_ago)
        except (ValueError, TypeError):
            pass

    # --- Priority 5: Date-based AGE calculation (if value is a year) ---
    if patient_data and 'age_at_visit' in patient_data and 'visit_date' in patient_data:
        try:
            age_at_visit = int(patient_data['age_at_visit'])
            visit_date = datetime.strptime(patient_data['visit_date'], '%Y-%m-%d')
            birth_year = visit_date.year - age_at_visit
            
            menopause_year = int(value_str.strip())
            if 1900 < menopause_year < 2100:
                menopause_age = menopause_year - birth_year
                return int(menopause_age)
        except (ValueError, TypeError):
             pass

    # If all else fails, return the translated text
    return translate_text(value_str)


def parse_menarche(value_str, patient_data=None):
    """
    Parses the 'Menarche' field, which can be an age, a year, or descriptive text.
    Calculates age if a year is provided.
    """
    if not value_str or value_str.strip() in ['-', '—', '--'] or not isinstance(value_str, str):
        return None

    value_str_lower = value_str.lower().strip()
    numbers = re.findall(r'\d+', value_str_lower)

    if not numbers:
        return translate_text(value_str)

    # Case 1: A single number is found.
    if len(numbers) == 1 or len(numbers) == 2:
        num = int(numbers[0])
        # If it's a small number, it's likely the age directly.
        if 5 < num < 25:
            return num
        # If it's a year, calculate the age at menarche.
        elif 1900 < num < 2100 and patient_data and 'age_at_visit' in patient_data and 'visit_date' in patient_data:
            try:
                age_at_visit = int(patient_data['age_at_visit'])
                visit_date = datetime.strptime(patient_data['visit_date'], '%Y-%m-%d')
                birth_year = visit_date.year - age_at_visit
                menarche_age = num - birth_year
                if 5 < menarche_age < 25:
                    return menarche_age
            except (ValueError, TypeError, KeyError):
                pass  # Fall through if calculation is not possible

    # Case 2: It's a range of years or other text.
    return translate_text(value_str)


def parse_personal_history(div, patient_data):
    """Parses the 'Personal history' (descripcion1) div."""
    data = {}
    if not isinstance(div, Tag):
        return data

    # Process each <p> tag as a potential key-value pair
    for p_tag in div.find_all('p'):
        # Ensure we are working with a Tag object
        if not isinstance(p_tag, Tag):
            continue

        # Skip the title paragraph by checking its class
        if p_tag.has_attr('class') and 'titulo' in p_tag['class']:
            continue

        # Get the full text from the <p> tag
        text = p_tag.get_text(strip=True)
        
        if ':' in text:
            parts = text.split(':', 1)
            key = parts[0].replace('-', '').strip()
            value = parts[1].strip()

            # If the value is just a hyphen, treat it as empty and skip.
            if value == '-':
                continue

            # Clean up key (e.g., "Family history?")
            if key.endswith('?'):
                key = key[:-1]

            # If value contains '::', it's a list of items.
            if '::' in value:
                items = [item.strip() for item in value.split('::') if item.strip()]
                translated_items = [translate_text(item) for item in items]
                final_value = ', '.join(translated_items)
            else:
                # Fallback for single values that might still have artifacts
                if value.endswith('::'):
                    value = value[:-2].strip()
                final_value = translate_text(value)

            # If value is empty after stripping, skip this field
            if not final_value:
                continue

            clean_key = re.sub(r'[^a-z0-9_]', '', key.lower().replace(' ', '_'))

            if not clean_key:
                continue

            if 'last_menstrual_period' in clean_key:
                # Pass the original value to the specialized parser
                lmp_value = parse_last_menstrual_period(value, patient_data)
                if isinstance(lmp_value, int) and lmp_value < 30:
                    patient_id = patient_data.get('patient_id', 'Unknown')
                    print(f"  -> WARNING: Patient {patient_id} has an unusually low last menstrual period age ({lmp_value}). Please verify.")
                data['last_menstrual_period'] = lmp_value
            elif 'menopause' in clean_key:
                # Pass the original value to the specialized parser
                data['menopause'] = parse_menopause(value, patient_data)
            elif 'menarche' in clean_key:
                menarche_value = parse_menarche(value, patient_data)
                if isinstance(menarche_value, int) and menarche_value > 17:
                    patient_id = patient_data.get('patient_id', 'Unknown')
                    print(f"  -> WARNING: Patient {patient_id} has an unusually high menarche age ({menarche_value}). Please verify.")
                data['menarche'] = menarche_value
            else:
                data[clean_key] = final_value
    return data


def parse_medical_history(div):
    """Parses the 'Medical history' (descripcion2) div, handling complex HTML."""
    data = {}
    if not isinstance(div, Tag):
        return data

    # The content is a mix of <p> tags, <span>s, and NavigableStrings.
    # We iterate through all children of the div to correctly associate keys and values.
    contents = div.contents
    i = 0
    while i < len(contents):
        item = contents[i]
        key, value = '', ''
        
        span = None
        # The key is usually in a <span> inside a <p> or just a <span>
        if isinstance(item, Tag):
            if item.name == 'p':
                span = item.find('span')
                if span:
                    key = span.get_text(strip=True)
                    # The value is the remaining text in the <p> tag
                    value = ''.join([s for s in item.strings if s != key]).strip()
            elif item.name == 'span':
                span = item
                key = span.get_text(strip=True)
                # The value might be in the *next* NavigableString sibling
                if not value and i + 1 < len(contents):
                    next_item = contents[i+1]
                    if isinstance(next_item, NavigableString):
                        value = next_item.strip()
                        if value:
                            # Consume the next item since we've used it
                            i += 1
        
        # Fallback for text nodes that start with a hyphen but aren't in a <p>
        elif isinstance(item, NavigableString) and item.strip().startswith('-'):
            text = item.strip().lstrip('-').strip()
            delimiter = '?' if '?' in text else ':'
            if delimiter in text:
                parts = text.split(delimiter, 1)
                key = parts[0]
                value = parts[1]

        if not key or not value:
            i += 1
            continue

        # Clean up key and value
        key = key.rstrip('?:').strip()
        value = value.lstrip(':- ').strip()

        # Handle typo "informationss"
        if 'informationss' in key:
            key = key.replace('informationss', 'informations')
            key = 'medical_' + key

        clean_key = re.sub(r'[^a-z0-9_]', '', key.lower().replace(' ', '_'))
        if clean_key:
            data[clean_key] = translate_text(value)
        
        i += 1
            
    return data


def parse_protocol_recommendations(div):
    """Parses the 'Protocol recommendations' (descripcion3) div."""
    data = {}
    if not isinstance(div, Tag):
        return data

    # Iterate through all direct children to handle mixed content,
    # without returning early if "no information provided" is found.
    for item in div.contents:
        text = ""
        if isinstance(item, NavigableString):
            text = item.strip()
        elif isinstance(item, Tag) and item.name == 'p':
            text = item.get_text(strip=True)
        
        # Skip irrelevant or empty lines
        if not text or "two hours ago patient" in text.lower() or "no information provided" in text.lower():
            continue

        # Handle "Body temperature: 36.50"
        if "body temperature" in text.lower() and ':' in text:
            parts = text.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip()
            clean_key = re.sub(r'[^a-z0-9_]', '', key.lower().replace(' ', '_'))
            if clean_key:
                data[clean_key] = value
            continue

        # Handle lines like "- Smoked? No"
        if text.startswith('-') and '?' in text:
            parts = text.lstrip('- ').split('?', 1)
            key = parts[0].strip()
            value = parts[1].strip()
            
            # Clean up key and create a standard prefix
            clean_key = "protocol_" + re.sub(r'[^a-z0-9_]', '', key.lower().replace(' ', '_'))
            
            if clean_key:
                data[clean_key] = translate_text(value)

    return data


def scrape_patient_data(session, patient_id, base_url, retries=3):
    """Scrapes a single patient page for all metadata with a retry mechanism."""
    target_url = f"{base_url}?id={patient_id}"
    
    for attempt in range(retries):
        try:
            response = session.get(target_url, timeout=15)
            response.raise_for_status()

            if "login.php" in response.url:
                print(f"Session expired for patient {patient_id}.")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            data = {'patient_id': patient_id}

            # --- 1. Parse Profile Info (`perfiluser`) ---
            perfil = soup.find('div', class_='perfiluser')
            if isinstance(perfil, Tag):
                record_p = perfil.find('p', class_='prontuario')
                if record_p:
                    data['record'] = record_p.get_text(strip=True).replace('Record:', '').strip()
                
                age_text_p = perfil.find('p', string=re.compile(r'years old'))
                if age_text_p:
                    match = re.search(r'(\w+), (\d+) years old', age_text_p.get_text(strip=True))
                    if match:
                        data['role'] = match.group(1)
                        age = int(match.group(2))
                        # Add age validation to skip unrealistic entries
                        if age > 100:
                            print(f"  -> WARNING: Patient {patient_id} has an unlikely age ({age}). Skipping patient.")
                            return None
                        data['age_current'] = age

                reg_date_p = perfil.find('p', string=re.compile(r'Registered at'))
                if reg_date_p:
                    reg_date_str = reg_date_p.get_text(strip=True).replace('Registered at', '').split('(y-m-d)')[0].strip()
                    data['registration_date'] = parse_date(reg_date_str).strftime('%Y-%m-%d')

                status_race_p = perfil.find('p', string=re.compile(r'Marital status:'))
                if status_race_p:
                    text = status_race_p.get_text(strip=True)
                    status_match = re.search(r'Marital status: ([\w\s]+)\.', text)
                    race_match = re.search(r'Race: (\w+)', text)
                    if status_match:
                        data['marital_status'] = translate_text(status_match.group(1).strip())
                    if race_match:
                        data['race'] = translate_text(race_match.group(1).strip())

                if 'age_current' in data and 'registration_date' in data:
                    year_current = datetime.now().year
                    year_visit = datetime.strptime(data['registration_date'], '%Y-%m-%d').year
                    data['age_at_visit'] = int(data['age_current'] - (year_current - year_visit))

            # --- 2. Parse Visit Date (from <label for="q2">) ---
            q2_label = soup.find('label', {'for': 'q2'})
            if q2_label:
                visit_date_text = q2_label.get_text(strip=True)
                if ':' in visit_date_text:
                    visit_date_str = visit_date_text.split(':', 1)[-1].strip()
                    try:
                        data['visit_date'] = parse_date(visit_date_str).strftime('%Y-%m-%d')
                    except (ParserError, TypeError):
                        data['visit_date'] = visit_date_str # save as is if parsing fails

            # --- 3. Parse Visit Info (`visitauser`) ---
            visita = soup.find('div', class_='visitauser')
            if isinstance(visita, Tag):
                diag_p = visita.find('p', class_='view-diagnostico')
                if isinstance(diag_p, Tag):
                    diag_span = diag_p.find('span')
                    if diag_span:
                        data['diagnosis'] = diag_span.get_text(strip=True)

                # Parse each description section with its specific parser
                for desc_div in visita.find_all('div', class_=re.compile(r'descripcion\d')):
                    if isinstance(desc_div, Tag):
                        section_class = desc_div.get('class')
                        section_data = {}
                        if section_class and 'descripcion1' in section_class:
                            section_data = parse_personal_history(desc_div, data)
                        elif section_class and 'descripcion2' in section_class:
                            section_data = parse_medical_history(desc_div)
                        elif section_class and 'descripcion3' in section_class:
                            section_data = parse_protocol_recommendations(desc_div)
                        
                        # Update data only with non-empty values to avoid overwriting
                        for key, value in section_data.items():
                            if value is not None and value != '':
                                data[key] = value
            
            # --- 4. Post-parsing verification and cleaning ---
            # Clear last menstrual period if patient is older and LMP is a recent date (data error)
            try:
                age_at_visit = data.get('age_at_visit')
                lmp_val = data.get('last_menstrual_period')

                # Check if age is high and LMP is a date string (e.g., 'YYYY-MM-DD')
                if age_at_visit and lmp_val and isinstance(lmp_val, str) and re.match(r'\d{4}-\d{2}-\d{2}', lmp_val):
                    if int(age_at_visit) > 60: # Threshold for menopause
                        patient_id = data.get('patient_id', 'Unknown')
                        print(f"  -> INFO: Patient {patient_id} (age {age_at_visit}) has a recent LMP date ({lmp_val}). Clearing field due to likely data error.")
                        data['last_menstrual_period'] = None
            except (ValueError, TypeError):
                pass # Ignore if values are not comparable

            # Swap Last Menstrual Period and Menopause if they seem inverted
            try:
                lmp_val = data.get('last_menstrual_period')
                meno_val = data.get('menopause')

                # Ensure both values are present and are numeric (i.e., ages)
                if lmp_val is not None and meno_val is not None:
                    lmp_age = int(lmp_val)
                    meno_age = int(meno_val)

                    # We should only compare if both values are plausible ages (e.g., < 100).
                    # This prevents comparing a year (e.g., 1998) with an age.
                    is_lmp_age_valid = 0 < lmp_age < 100
                    is_meno_age_valid = 0 < meno_age < 100

                    # If both are valid ages and LMP age is greater than menopause age, they were likely swapped.
                    if is_lmp_age_valid and is_meno_age_valid and lmp_age > meno_age:
                        patient_id = data.get('patient_id', 'Unknown')
                        print(f"  -> INFO: Patient {patient_id}: Swapping Last Menstrual Period ({lmp_age}) and Menopause ({meno_age}) due to logical inconsistency.")
                        data['last_menstrual_period'], data['menopause'] = meno_age, lmp_age

            except (ValueError, TypeError):
                # This will happen if one or both fields are not age-like numbers (e.g., a date, 'Yes', 'No'),
                # in which case we don't perform the swap.
                pass

            return data

        except requests.exceptions.RequestException as e:
            print(f"  -> Attempt {attempt + 1} failed for patient {patient_id}: {e}")
            if attempt < retries - 1:
                time.sleep((attempt + 1) * 2)
            else:
                print(f"  -> All retries failed for patient {patient_id}.")
                return None
        except Exception as e:
            print(f"  -> Error parsing data for patient {patient_id}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Web scrap patient's details on DMR-IR to get complete metadata."
    )
    parser.add_argument("--output", type=str, default="patient_metadata.csv", help="Output CSV file path.")
    parser.add_argument("--start_id", type=int, default=1, help="Starting patient ID.")
    parser.add_argument("--end_id", type=int, default=425, help="Ending patient ID.")
    parser.add_argument("--num_patients", type=int, help="Number of random patients to scrape.")
    args = parser.parse_args()

    LOGIN_URL = "https://visual.ic.uff.br/dmi/login.php"
    BASE_DETAILS_URL = "https://visual.ic.uff.br/dmi/prontuario/details.php"
    
    username = "guillepinto"
    password = "~y5nz+YEKBzx7&J"
    
    all_data = []
    with requests.Session() as session:
        login_payload = {'usuario': username, 'password': password}
        
        print(f"Attempting to log in to {LOGIN_URL}...")
        try:
            login_response = session.post(LOGIN_URL, data=login_payload)
            login_response.raise_for_status()
            
            if "login.php" in login_response.url:
                print("\nLogin failed. Please check credentials.")
                return

            print("Login successful.")
        except requests.exceptions.RequestException as e:
            print(f"\nLogin request failed: {e}")
            return

        patient_ids = list(range(args.start_id, args.end_id + 1))
        if args.num_patients:
            if args.num_patients > len(patient_ids):
                print(f"Warning: Requested {args.num_patients} patients, but the range only contains {len(patient_ids)}.")
            else:
                patient_ids = random.sample(patient_ids, args.num_patients)

        for patient_id in tqdm(patient_ids, desc="Scraping Patient Pages"):
            patient_data = scrape_patient_data(session, patient_id, BASE_DETAILS_URL)
            if patient_data:
                all_data.append(patient_data)
            time.sleep(random.uniform(0.5, 1.5))

    if not all_data:
        print("\nNo data was scraped. Exiting.")
        return

    output_csv = args.output
    df = pd.DataFrame(all_data)
    
    # Convert float columns to nullable integer where possible
    for col in ['age_at_visit', 'age_current', 'menarche']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # Reorder columns for clarity, putting the most important ones first
    preferred_cols = [
        'patient_id', 'record', 'role', 'age_current','age_at_visit',  
        'registration_date', 'marital_status', 'race', 'visit_date', 'diagnosis',
        'complaints', 'symptoms', 'signs', 'last_menstrual_period', 'menopause',
        'menarche', 'eating_habits', 'cancer_family', 'family_history',
        'further_informations', 'mammography', 'radiotherapy', 'plastic_surgery',
        'prosthesis', 'biopsy', 'use_of_hormone_replacement', 'nipple_changes',
        'is_there_signal_of_wart_on_breast', 'medical_further_informations',
        'body_temperature', 'protocol_smoked', 'protocol_drank_coffee',
        'protocol_consumed_alcohol', 'protocol_physical_exercise',
        'protocol_put_some_pomade_deodorant_or_products_at_breasts_or_armpits_region'
    ]
    
    existing_preferred_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = sorted([c for c in df.columns if c not in existing_preferred_cols])
    
    df = df[existing_preferred_cols + other_cols]

    df.to_csv(output_csv, index=False)
    
    print(f"\nScraping complete. {len(df)} records saved to {output_csv}")
    print("\nSample of the data:")

    # Print a detailed summary of each patient's data
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"\n\n🩺 Patient Record {i + 1}\n{'='*40}")
        # Convert row to a dictionary, handling potential NaNs
        row_dict = {k: ('' if pd.isna(v) else v) for k, v in row.to_dict().items()}
        
        # Format the dictionary into a two-column table
        table_data = [[k, v] for k, v in row_dict.items()]
        print(tabulate(table_data, headers=["Field", "Value"], tablefmt="grid"))


if __name__ == "__main__":
    main()