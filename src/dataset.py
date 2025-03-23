import os
import json
from torch.utils.data import Dataset
from src.single_prompt import convert_json_to_sentence
from src.category_prompt import generate_category_prompts

class PromptDataset(Dataset):
    def __init__(self, folder_path, labels_path, 
                 include_demographic=True, include_personal_history=True, 
                 include_medical_history=True, include_protocol=True):
        # List JSON files in the folder
        self.json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]
        self.include_demographic = include_demographic
        self.include_personal_history = include_personal_history
        self.include_medical_history = include_medical_history
        self.include_protocol = include_protocol

        # Load patient labels and create a mapping patient_id -> binary_label
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        self.labels_dict = { str(int(float(label["patient_id"]))): label["binary_label"] for label in labels }

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        with open(json_file, 'r') as f:
            data = json.load(f)
        # Create prompt from JSON using the conversion function
        prompt = convert_json_to_sentence(
            data,
            include_demographic=self.include_demographic,
            include_personal_history=self.include_personal_history,
            include_medical_history=self.include_medical_history,
            include_protocol=self.include_protocol
        )
        patient_id = str(int(float(data["id_paciente"])))
        label = self.labels_dict.get(patient_id, None)
        sample = {"prompt": prompt, "label": label}
        return sample

# New dataset class for multiple category prompts per patient
class MultiPromptDataset(Dataset):
    def __init__(self, folder_path, labels_path):
        # List JSON files in the folder
        self.json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]
        # Load patient labels and create a mapping patient_id -> binary_label
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        self.labels_dict = { str(int(float(label["patient_id"]))): label["binary_label"] for label in labels }

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        with open(json_file, 'r') as f:
            data = json.load(f)
        # Generate category prompts (risk factors, complementary features, protocol features)
        prompts = generate_category_prompts(data)
        patient_id = str(int(float(data["id_paciente"])))
        label = self.labels_dict.get(patient_id, None)
        sample = {"prompts": prompts, "label": label}
        return sample
