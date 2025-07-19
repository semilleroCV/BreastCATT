import torch
import torch.nn as nn

class CNNFront(nn.Module):
    def __init__(self, in_channels=3):
        super(CNNFront, self).__init__()
        self.conv_part = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_part(x)

class CNN90(nn.Module):
    def __init__(self, in_channels=3):
        super(CNN90, self).__init__()
        self.conv_part = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_part(x)
    
class FullyConnectedPart(nn.Module):
    def __init__(self, in_features):
        super(FullyConnectedPart, self).__init__()
        self.fc_part = nn.Sequential(
            # Block 1
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),

            # Block 2
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),

            # Final layers
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc_part(x)

class SanchezCauceNet(nn.Module):
    def __init__(self, input_shape=(224, 224), in_channels=3, use_clinical_data:bool=False):
        super(SanchezCauceNet, self).__init__()
        
        self.use_clinical_data = use_clinical_data

        self.cnn_front = CNNFront(in_channels)
        self.cnn_l90 = CNN90(in_channels)
        self.cnn_r90 = CNN90(in_channels)

        # Assuming 224x224 input, the feature sizes are fixed
        self.front_features = 100352 # 128 * (224/8) * (224/8)
        self.l90_features = 100352   # 128 * (224/8) * (224/8)
        self.r90_features = 100352   # 128 * (224/8) * (224/8)

        # Clinical branch when data is available
        if self.use_clinical_data:
            self.clinical_branch = FullyConnectedPart(18) # as stated in the original paper

        self.fc_front = FullyConnectedPart(self.front_features)
        self.fc_l90 = FullyConnectedPart(self.l90_features)
        self.fc_r90 = FullyConnectedPart(self.r90_features)

        self.final_layer = nn.Linear(4, 2) if self.use_clinical_data else nn.Linear(3, 2)

    def forward(self, x_front, x_l90, x_r90, clinical_data=None):
        # Front branch
        out_front = self.cnn_front(x_front)
        out_front = out_front.view(out_front.size(0), -1)
        prob_front = self.fc_front(out_front)

        # L90 branch
        out_l90 = self.cnn_l90(x_l90)
        out_l90 = out_l90.view(out_l90.size(0), -1)
        prob_l90 = self.fc_l90(out_l90)

        # R90 branch
        out_r90 = self.cnn_r90(x_r90)
        out_r90 = out_r90.view(out_r90.size(0), -1)
        prob_r90 = self.fc_r90(out_r90)

        # Concatenate probabilities
        if self.use_clinical_data and clinical_data is not None:
            out_clinical_branch = self.clinical_branch(clinical_data)
            concatenated_probs = torch.cat((prob_front, prob_l90, prob_r90,
                                            out_clinical_branch), dim=1)
        else:
            concatenated_probs = torch.cat((prob_front, prob_l90, prob_r90), dim=1)

        return self.final_layer(concatenated_probs) # output logits

def create_sanchez_cauce_net(use_clinical_data=False, pretrained_weights_path=None, freeze_vision_part=False, **kwargs):
    """
    Factory function to create the SanchezCauceNet model.

    Args:
        use_clinical_data (bool): If True, the model will include the clinical data branch.
        pretrained_weights_path (str, optional): Path to the pre-trained model weights.
        freeze_vision_part (bool): If True, freezes the weights of the vision parts of the model.
        **kwargs: Additional arguments for the SanchezCauceNet constructor.

    Returns:
        SanchezCauceNet: The created model.
    """
    model = SanchezCauceNet(use_clinical_data=use_clinical_data, **kwargs)

    if pretrained_weights_path:
        # Load pretrained weights, ignoring the final layer if architectures mismatch
        pretrained_dict = torch.load(pretrained_weights_path)
        model_dict = model.state_dict()
        
        # Filter out unnecessary keys (e.g., final_layer from a different architecture)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'final_layer' not in k}
        
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        model.load_state_dict(model_dict)

    if freeze_vision_part:
        for param in model.cnn_front.parameters():
            param.requires_grad = False
        for param in model.cnn_l90.parameters():
            param.requires_grad = False
        for param in model.cnn_r90.parameters():
            param.requires_grad = False
        for param in model.fc_front.parameters():
            param.requires_grad = False
        for param in model.fc_l90.parameters():
            param.requires_grad = False
        for param in model.fc_r90.parameters():
            param.requires_grad = False

    return model

if __name__ == '__main__':
    # Example usage:
    # Assuming input images are 224x224 RGB images
    input_shape = (224, 224)
    batch_size = 4
    
    model = SanchezCauceNet(input_shape=input_shape, use_clinical_data=True)
    
    # Create dummy inputs
    front_view = torch.randn(batch_size, 3, *input_shape)
    l90_view = torch.randn(batch_size, 3, *input_shape)
    r90_view = torch.randn(batch_size, 3, *input_shape)
    clinical_data = torch.randn(batch_size, 18)
    
    # Forward pass
    output = model(front_view, l90_view, r90_view, clinical_data=clinical_data)
    
    print("Model created successfully!")
    print(f"Input shapes: {front_view.shape}, {l90_view.shape}, {r90_view.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    # Print model summary
    print("\nModel Architecture:")
    print(model)

    # --- Factory function examples ---
    print("\n--- Factory Function Examples ---")

    # 1. Create a model for initial training (vision part only)
    print("\n1. Creating model for vision-only training...")
    vision_model = create_sanchez_cauce_net(use_clinical_data=False, input_shape=input_shape)
    print(vision_model)
    # Save dummy weights for demonstration
    torch.save(vision_model.state_dict(), 'vision_only_weights.pth')
    print("Saved dummy vision-only weights to 'vision_only_weights.pth'")


    # 2. Create a model with clinical data, loading pretrained vision weights and freezing them
    print("\n2. Creating model with clinical data, loading pretrained weights and freezing vision part...")
    full_model_frozen = create_sanchez_cauce_net(
        use_clinical_data=True,
        input_shape=input_shape,
        pretrained_weights_path='vision_only_weights.pth',
        freeze_vision_part=True
    )
    print(full_model_frozen)

    # Verify that vision parts are frozen
    vision_frozen = all(
        not p.requires_grad for p in list(full_model_frozen.cnn_front.parameters()) +
        list(full_model_frozen.fc_r90.parameters())
    )
    clinical_trainable = all(
        p.requires_grad for p in full_model_frozen.clinical_branch.parameters()
    )
    final_layer_trainable = all(
        p.requires_grad for p in full_model_frozen.final_layer.parameters()
    )

    print(f"\nVision part frozen: {vision_frozen}")
    print(f"Clinical branch trainable: {clinical_trainable}")
    print(f"Final layer trainable: {final_layer_trainable}")

    # Clean up the dummy weights file
    import os
    os.remove('vision_only_weights.pth')
