#!/usr/bin/env python3
"""
Test script to verify that train.py works with automatic MAE download functionality.
"""

import sys
import os
import subprocess

# Add the project root to Python path
sys.path.append(os.path.abspath("../"))

def test_train_with_automatic_download():
    """Test that train.py works with automatic MAE download."""
    print("🧪 Testing train.py with automatic MAE download functionality...")
    
    # Test command that should work with automatic download
    test_cmd = [
        'python', 'train.py',
        '--dataset_name', 'SemilleroCV/DMR-IR', 
        '--vit_version', 'base',
        '--per_device_train_batch_size', '1',
        '--per_device_eval_batch_size', '1', 
        '--num_train_epochs', '1',
        '--max_train_steps', '2',  # Very short test
        '--learning_rate', '1e-5',
        '--output_dir', './test_output',
        '--use_cross_attn', 'false',
        '--use_segmentation', 'false',
        '--checkpointing_steps', '1',
        '--with_tracking', 'false'  # Disable wandb tracking for testing
    ]
    
    print(f"Running command: {' '.join(test_cmd)}")
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            test_cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minute timeout
        )
        
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
            
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        # Check if automatic download message appears
        if "MAE base weights already exist" in result.stderr or "Downloading MAE base weights" in result.stderr:
            print("✅ Automatic MAE download functionality is working in train.py")
        else:
            print("⚠️  Could not confirm automatic download functionality")
            
        if result.returncode == 0:
            print("✅ train.py completed successfully")
        else:
            print(f"❌ train.py failed with return code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
    finally:
        # Clean up test output directory
        if os.path.exists('./test_output'):
            import shutil
            shutil.rmtree('./test_output')
            print("🧹 Cleaned up test output directory")

if __name__ == "__main__":
    test_train_with_automatic_download()
