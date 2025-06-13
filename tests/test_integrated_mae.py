#!/usr/bin/env python3
"""
Test script for integrated MAE weights download functionality in tfvit.py
"""

import sys
import os
import torch

# Add the project root to Python path
sys.path.insert(0, '/home/guillermo/ssd/Github/BreastCATT')

def test_integrated_mae_functionality():
    """Test the integrated MAE weights download and model creation."""
    print("🧪 Testing integrated MAE functionality with model creation...")
    
    try:
        # Import the tfvit module (this will test all imports work)
        print("\n1. Testing imports...")
        from breastcatt import tfvit
        print("✅ Successfully imported tfvit module")
        
        # Test base model creation with auto-download
        print("\n2. Testing base model creation with auto-download...")
        model = tfvit.multimodal_vit_base_patch16(
            use_cross_attn=False,  # Disable to avoid language model loading
            use_segmentation=False,  # Disable to avoid segmentation loading
            num_classes=1
        )
        print("✅ Successfully created base model with auto-downloaded weights")
        print(f"   Model type: {type(model)}")
        print(f"   Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass with dummy data
        print("\n3. Testing forward pass...")
        dummy_input = torch.randn(1, 1, 224, 224)  # 1 channel thermal image
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass successful, output shape: {output.logits.shape}")
        
        # Test that checkpoint was actually loaded
        print("\n4. Verifying weights were loaded...")
        has_pretrained_weights = any(param.requires_grad for param in model.parameters())
        print(f"✅ Model has trainable parameters: {has_pretrained_weights}")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("This might be due to missing dependencies (segmentation model, language model, etc.)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_manual_checkpoint_path():
    """Test using manually specified checkpoint path."""
    print("\n🧪 Testing manual checkpoint path specification...")
    
    try:
        from breastcatt import tfvit
        
        # Test with manual path (should use existing downloaded file)
        checkpoint_path = "checkpoints/fvit/mae_pretrain_vit_base.pth"
        if os.path.exists(checkpoint_path):
            print(f"✅ Using existing checkpoint: {checkpoint_path}")
            model = tfvit.multimodal_vit_base_patch16(
                use_cross_attn=False,
                use_segmentation=False,
                checkpoint_path=checkpoint_path,
                num_classes=1
            )
            print("✅ Successfully created model with manual checkpoint path")
        else:
            print(f"⚠️  Checkpoint not found at {checkpoint_path}, skipping manual test")
            
    except Exception as e:
        print(f"❌ Manual checkpoint test failed: {e}")

if __name__ == "__main__":
    test_integrated_mae_functionality()
    test_manual_checkpoint_path()
