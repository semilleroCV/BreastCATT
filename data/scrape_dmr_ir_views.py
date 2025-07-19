import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from tqdm import tqdm
import random
import time
import argparse

def scrape_patient_data(session, patient_id, base_url):
    """Scrapes a single patient page for image and view information."""
    target_url = f"{base_url}?id={patient_id}"
    patient_image_data = []
    
    try:
        response = session.get(target_url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Check if we were redirected to the login page, which means the session is invalid
        if "login.php" in response.url or "index.php" in response.url:
            print(f"Session expired or invalid when accessing patient {patient_id}. Please check login.")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        
        image_container = soup.find('div', class_='imagenspaciente')
        
        if not image_container:
            return patient_image_data # Return empty list if no images found

        image_links = image_container.find_all('a', class_='figura')
        
        image_index = 1
        for link in image_links:
            image_href = link.get('href')
            image_title = link.get('title')
            
            if image_href and image_title:
                if image_title == "Maximizar":
                    continue
                else:
                    image_filename = os.path.basename(image_href)
                    formatted_index = f"{image_index:03d}"
                    patient_image_data.append({
                        'patient_id_scraped': patient_id,
                        'image_filename': image_filename,
                        'view': image_title,
                        'image_index': formatted_index
                    })
                    image_index += 1
        return patient_image_data

    except requests.exceptions.RequestException as e:
        print(f"  -> Error fetching page for patient {patient_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Web scrap patient's details on DMR-IR to get the view's names."
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path.")
    args = parser.parse_args()

    # --- Configuration ---
    # The form's 'action' attribute is 'login.php', so we must POST to that URL.
    LOGIN_URL = "https://visual.ic.uff.br/dmi/login.php"
    BASE_DETAILS_URL = "https://visual.ic.uff.br/dmi/prontuario/details.php"
    
    # --- User Input ---
    username = "guillepinto"
    password = "~y5nz+YEKBzx7&J" # ~y5nz+YEKBzx7&J
    
    start_id = 1
    end_id = 425

    # --- Main Logic ---
    all_data = []
    with requests.Session() as session:
        # NOTE: You may need to inspect the login form's HTML to find the correct
        # names for the username and password fields (e.g., 'username', 'user', 'login', 'pass', 'senha').
        login_payload = {
            'usuario': username,
            'password': password
        }
        
        print(f"Attempting to log in to {LOGIN_URL}...")
        try:
            login_response = session.post(LOGIN_URL, data=login_payload)
            login_response.raise_for_status()
            
            if login_response.url == "https://visual.ic.uff.br/dmi/" or login_response.url == "https://visual.ic.uff.br/dmi/index.php":
                print("\nLogin failed: The server returned to the login page. Please check credentials.")
                return

            print("Login successful.")
            # This is useful for debugging to see where you landed.
            print(f"Landed on page: {login_response.url}")

        except requests.exceptions.RequestException as e:
            print(f"\nLogin request failed: {e}")
            return

        # Loop through all patient IDs
        for patient_id in tqdm(range(start_id, end_id + 1), desc="Scraping Patient Pages"):
            patient_data = scrape_patient_data(session, patient_id, BASE_DETAILS_URL)
            if patient_data:
                all_data.extend(patient_data)
            time.sleep(random.uniform(1, 2))

    # --- Save to CSV ---
    if not all_data:
        print("\nNo data was scraped. Exiting.")
        return

    output_csv = args.output
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    
    print(f"\nScraping complete. {len(df)} records saved to {output_csv}")
    print("\nSample of the data:")
    print(df.head())

if __name__ == "__main__":
    main()