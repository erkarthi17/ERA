"""
Test script to verify Hugging Face ImageNet setup works.
This script tests data loading without full training.
"""

import torch
from config import Config
from data_hf import get_data_loaders, load_imagenet_from_hf
from model import resnet50
import time

def test_huggingface_login():
    """Test if user is logged in to Hugging Face."""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"Logged in to Hugging Face as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"Not logged in to Hugging Face: {e}")
        print("Please run: huggingface-cli login")
        return False

def test_imagenet_loading():
    """Test loading ImageNet dataset."""
    print("\n" + "="*50)
    print("Testing ImageNet dataset loading...")
    print("="*50)
    
    try:
        # Test with small subsets
        train_subset = 100  # Small subset for testing
        val_subset = 50
        
        train_dataset, val_dataset = load_imagenet_from_hf(
            train_subset_size=train_subset,
            val_subset_size=val_subset
        )
        
        print(f"Successfully loaded ImageNet subsets:")
        print(f"   Train subset: {len(train_dataset)} samples")
        print(f"   Val subset: {len(val_dataset)} samples")
        
        # Test a few samples
        print(f"\nTesting sample access...")
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            print(f"   Sample {i}: image shape={sample['image'].size}, label={sample['label']}")
        
        return True
        
    except Exception as e:
        print(f"Error loading ImageNet: {e}")
        return False

def test_data_loaders():
    """Test data loader creation."""
    print("\n" + "="*50)
    print("Testing data loader creation...")
    print("="*50)
    
    try:
        # Create config with small subsets
        config = Config()
        config.train_subset_size = 100
        config.val_subset_size = 50
        config.batch_size = 16
        config.num_workers = 0  # Avoid multiprocessing issues in testing
        
        train_loader, val_loader = get_data_loaders(config)
        
        print(f"Successfully created data loaders:")
        print(f"   Train loader: {len(train_loader)} batches, {len(train_loader.dataset)} samples")
        print(f"   Val loader: {len(val_loader)} batches, {len(val_loader.dataset)} samples")
        
        # Test loading a batch
        print(f"\nTesting batch loading...")
        start_time = time.time()
        
        for i, (images, labels) in enumerate(train_loader):
            print(f"   Batch {i}: images={images.shape}, labels={labels.shape}")
            print(f"   Image dtype: {images.dtype}, Label dtype: {labels.dtype}")
            print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
            if i >= 2:  # Test only first 3 batches
                break
        
        load_time = time.time() - start_time
        print(f"Batch loading successful (took {load_time:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_forward():
    """Test model forward pass."""
    print("\n" + "="*50)
    print("Testing model forward pass...")
    print("="*50)
    
    try:
        # Create model
        model = resnet50(num_classes=1000)
        model.eval()
        
        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        # Test forward pass
        with torch.no_grad():
            start_time = time.time()
            output = model(dummy_input)
            forward_time = time.time() - start_time
        
        print(f"Model forward pass successful:")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Forward time: {forward_time:.3f}s")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"Error in model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing Hugging Face ImageNet Setup")
    print("="*60)
    
    tests = [
        ("Hugging Face Login", test_huggingface_login),
        ("ImageNet Loading", test_imagenet_loading),
        ("Data Loaders", test_data_loaders),
        ("Model Forward Pass", test_model_forward)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"{test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\nAll tests passed! You're ready to train with Hugging Face ImageNet.")
        print("\nNext steps:")
        print("1. Run: python train_hf.py --train-subset 1000 --val-subset 500 --epochs 1")
        print("2. For full training: python train_hf.py --epochs 90")
    else:
        print("\nSome tests failed. Please fix the issues before training.")
        print("\nCommon fixes:")
        print("- Run: huggingface-cli login")
        print("- Ensure you have access to ImageNet dataset")
        print("- Check internet connection")

if __name__ == "__main__":
    main()