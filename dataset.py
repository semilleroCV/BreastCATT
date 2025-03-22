import os
import json
from torch.utils.data import Dataset
from utils import convert_json_to_sentence, generate_category_prompts

class PromptDataset(Dataset):
    def __init__(self, folder_path, labels_path, transform=None):
        # List JSON files in the folder
        self.json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]
        self.transform = transform

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
        prompt = convert_json_to_sentence(data)
        patient_id = str(int(float(data["id_paciente"])))
        label = self.labels_dict.get(patient_id, None)
        sample = {"prompt": prompt, "label": label}
        if self.transform:
            sample = self.transform(sample)
        return sample

# New dataset class for multiple category prompts per patient
class MultiPromptDataset(Dataset):
    def __init__(self, folder_path, labels_path, transform=None):
        # List JSON files in the folder
        self.json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]
        self.transform = transform
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
        if self.transform:
            sample = self.transform(sample)
        return sample
