import argparse
import sys
import os
import numpy as np
import torch

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset
from breastcatt.segmenter import SegmentationModel
from breastcatt.tfvit_precomputed import download_transunet_weights
from torchvision.transforms import Compose, Resize, ToTensor, Lambda

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute segmentation masks and upload them to Hugging Face.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SemilleroCV/DMR-IR",
        help="The name of the dataset on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--image_column_name",
        type=str,
        default="image",
        help="The name of the column containing the images.",
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="train",
        help="The split of the dataset to process (e.g., 'train', 'test')."
    )
    parser.add_argument(
        "--new_config_name",
        type=str,
        default=None,
        help="The name for the new dataset configuration on the Hub."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Segmentation Model
    print("Loading segmentation model...")
    segmentation_weights_path = download_transunet_weights()
    segmentation_model = SegmentationModel(
        img_size=224, n_skip=3, num_classes=1,
        dir_model=segmentation_weights_path,
        device=torch.device(device)
    )

    # 2. Load Dataset (assuming it already has embeddings)
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split=args.split)

    # 3. Define Transforms and Computation Function
    # Transforms required by the segmentation model

    MAX_TEMPERATURE = 36.44

    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),  # Converts to [0, 1] float tensor from [0, 255] PIL image
        Lambda(lambda x: x / MAX_TEMPERATURE)  # Normalize by MAX_TEMPERATURE
    ])

    def compute_segmentation_mask(batch):
      # images = [img.convert("RGB") for img in batch[args.image_column_name]]
      # pixel_values = torch.stack([transforms(image) for image in images]).to(device)
      pixel_values = torch.stack([transforms(image) for image in batch[args.image_column_name]]).to(device)
      
      with torch.no_grad():
          logits = segmentation_model(pixel_values)
          # Create a binary mask (0 or 1)
          seg_mask = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
          
      # The map function expects a list of arrays
      return {"segmentation_mask": [mask for mask in seg_mask]}
    
    # 4. Map the function over the dataset
    print("Computing segmentation masks...")
    dataset_with_masks = dataset.map(
        compute_segmentation_mask, 
        batched=True, 
        batch_size=16,
        desc="Generating segmentation masks"
    )

    # 5. Push the new dataset to the Hub
    if args.new_config_name:
        print(f"Pushing new dataset to the Hub under config: {args.new_config_name}")
    else:
        print("Pushing updated dataset to the Hub (default config)...")

    # Make sure you are logged in: `huggingface-cli login`
    # push_to_hub will use the default config if args.new_config_name is None
    dataset_with_masks.push_to_hub(args.dataset_name, config_name=args.new_config_name)
    
    print("Done!")

if __name__ == "__main__":
    main()