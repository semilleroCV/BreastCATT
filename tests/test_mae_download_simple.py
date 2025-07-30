#!/usr/bin/env python3
"""
Simple test script for MAE weights download functionality (bypassing segmentation imports).
"""

import sys
import os
import urllib.request

# Add the project root to Python path
sys.path.append(os.path.abspath("../"))

def download_mae_weights(model_size="base", force_download=False):
    """
    Download MAE pretrained weights from Facebook's servers.
    
    Args:
        model_size (str): Size of the model ("base" or "large")
        force_download (bool): Whether to force re-download if file exists
        
    Returns:
        str: Path to the downloaded checkpoint file
    """
    # Define download URLs for different model sizes
    mae_urls = {
        "base": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        "large": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth"
    }
    
    if model_size not in mae_urls:
        raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(mae_urls.keys())}")
    
    # Create checkpoints directory
    checkpoint_dir = "checkpoints/fvit"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define local file path
    filename = f"mae_pretrain_vit_{model_size}.pth"
    local_path = os.path.join(checkpoint_dir, filename)
    
    # Check if file already exists
    if os.path.exists(local_path) and not force_download:
        print(f"✅ MAE {model_size} weights already exist at: {local_path}")
        return local_path
    
    # Download the file
    url = mae_urls[model_size]
    print(f"📥 Downloading MAE {model_size} weights from Facebook's servers...")
    print(f"URL: {url}")
    print(f"Saving to: {local_path}")
    
    try:
        # Download with progress indication
        def download_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100.0, block_num * block_size * 100.0 / total_size)
                print(f"\rProgress: {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, local_path, download_progress)
        print(f"\n✅ Successfully downloaded MAE {model_size} weights to: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"\n❌ Error downloading MAE weights: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)  # Clean up partial download
        raise

def test_download_functionality():
    """Test the MAE weights download functionality."""
    print("🧪 Testing MAE weights download functionality...")
    
    # Test base model download (just a small test to avoid large downloads)
    try:
        print("\n1. Testing base model download...")
        checkpoint_path = download_mae_weights("base")
        print(f"✅ Base model checkpoint available at: {checkpoint_path}")
        
        # Check if file exists and has reasonable size
        if os.path.exists(checkpoint_path):
            file_size = os.path.getsize(checkpoint_path)
            print(f"📁 File size: {file_size / (1024*1024):.1f} MB")
            if file_size > 100 * 1024 * 1024:  # > 100MB
                print("✅ File size looks reasonable for a base model")
            else:
                print("⚠️  File size seems small, download might be incomplete")
        else:
            print("❌ File does not exist after download")
            
    except Exception as e:
        print(f"❌ Base model download failed: {e}")
    
    # Test invalid model size
    try:
        print("\n2. Testing invalid model size...")
        download_mae_weights("invalid")
        print("❌ Should have raised an error for invalid model size")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_download_functionality()
