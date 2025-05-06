import torch
import torch.nn as nn
from transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from transunet.vit_seg_modeling import CONFIGS

class SegmentationModel(nn.Module):
    def __init__(self, img_size: int, n_skip: int, num_classes: int, dir_model: str, device: torch.device, threshold: float = 0.5):
        """
        Initializes the segmentation model with a Vision Transformer (ViT) backbone.
        
        Args:
            img_size (int): The size of the input images.
            n_skip (int): The number of skip connections.
            num_classes (int): Number of segmentation classes.
            dir_model (str): Path to the model weights file.
            device (torch.device): The device on which to load the model (CPU or GPU).
            threshold (float, optional): Threshold for converting probabilities into binary predictions. Defaults to 0.5.
        """
        super().__init__()

        self.device = device
        self.threshold = threshold

        # Fixed configuration; you can try other configurations like "ViT-B_16".
        self.config_vit = CONFIGS["R50-ViT-B_16"]  # You can try others like "ViT-B_16"
        self.config_vit.n_classes = num_classes  # Number of classes for binary segmentation
        self.config_vit.n_skip = n_skip
        self.config_vit.patches.grid = (14, 14)

        # Initialize the segmentation model using ViT_seg
        self.model = ViT_seg(self.config_vit, img_size=img_size, num_classes=num_classes).to(device)

        # Load model weights from file, mapping to the specified device
        try:
            self.model.load_state_dict(torch.load(dir_model, map_location=device, weights_only=True))
        except Exception as e:
            print(f"Error loading weights from {dir_model}: {e}")
            raise
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def predict_on_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs prediction on a single image.
        
        Args:
            image_tensor (torch.Tensor): Input image tensor expected to have shape [1, C, H, W].
        
        Returns:
            torch.Tensor: Binary segmentation prediction with shape [C, H, W].
        """
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)  # Move the tensor to the appropriate device
            logits = self.model(image_tensor)             # Forward pass; output shape: [1, C, H, W]
            probs = torch.sigmoid(logits)                   # Convert logits to probabilities using sigmoid

            # Convert probabilities to binary mask using the defined threshold
            preds = (probs > self.threshold).float()
            return preds