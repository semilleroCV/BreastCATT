import torch
import torch.nn as nn
import clip

class ClipLinearProbing(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # Assume CLIP text encoder outputs 512-dim embeddings (adjust if different)

    def forward(self, prompts: list):
        # Tokenize the input prompts
        tokens = clip.tokenize(prompts, truncate=True).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(tokens)
        return text_embeddings
