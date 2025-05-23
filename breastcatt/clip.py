import torch
import torch.nn as nn
import clip
from transformers import CLIPTokenizer, CLIPModel

class ClipOpenai(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, prompts: list):
        # Tokenize the input prompts
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(tokens)
        return text_embeddings
    
class ClipTransformers(nn.Module):

    def __init__(self, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_model.eval()
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, prompts: list):
        # Tokenize the input prompts
        tokens = self.tokenizer(prompts, return_tensors="pt", max_length=77, truncation=True).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**tokens)
        return text_embeddings
