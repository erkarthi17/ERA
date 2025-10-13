"""
Quick test script to verify the setup is working correctly.
Run this before starting full training.
"""

import torch
import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import numpy as np
        from model import resnet50
        from config import Config
        from data import get_data_loaders
        from utils import accuracy, count_parameters
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_model():
    """Test if ResNet-50 model can be created."""
    print("\nTesting model creation...")
    try:
        from model import resnet50
        model = resnet50(num_classes=1000)
        num_params = count_parameters(model)
        print(f"✅ ResNet-50 created successfully")
        print(f"   Parameters: {num_params:,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    try:
        from config import Config
        config = Config()
        print("✅ Configuration loaded")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Epochs: {config.epochs}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_data_loading():
    """Test data loading (will fail if ImageNet path is not set)."""
    print("\nTesting data loading...")
    try:
        from config import Config
        from data import get_data_loaders
        
        config = Config()
        
        # Check if data path exists
        if not os.path.exists(config.train_path):
            print(f"⚠️  Training data path does not exist: {config.train_path}")
            print("   Please update config.py with correct ImageNet path")
            return False
        
        # Try to create data loaders
        train_loader, val_loader = get_data_loaders(config)
        print("✅ Data loaders created successfully")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
        
        # Test loading a batch
        for images, labels in train_loader:
            print(f"   Batch shape: {images.shape}")
            print(f"   Labels shape: {labels.shape}")
            break
        
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        print("   Make sure ImageNet dataset is properly set up")
        return False


def test_accuracy_function():
    """Test accuracy calculation."""
    print("\nTesting accuracy function...")
    try:
        from utils import accuracy
        
        # Create dummy predictions and targets
        output = torch.randn(32, 1000)
        target = torch.randint(0, 1000, (32,))
        
        top1, top5 = accuracy(output, target)
        print(f"✅ Accuracy calculation works")
        print(f"   Top-1: {top1.item():.2f}%")
        print(f"   Top-5: {top5.item():.2f}%")
        return True
    except Exception as e:
        print(f"❌ Accuracy calculation error: {e}")
        return False


def test_device():
    """Test GPU availability."""
    print("\nTesting device...")
    try:
        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available, will use CPU")
            print("   Training will be significantly slower")
        return True
    except Exception as e:
        print(f"❌ Device check error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("ResNet-50 Setup Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Model", test_model),
        ("Accuracy Function", test_accuracy_function),
        ("Device", test_device),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to start training.")
        print("\nNext steps:")
        print("1. Ensure ImageNet dataset is properly set up")
        print("2. Update config.py with correct data path")
        print("3. Run: python train.py --data-root /path/to/imagenet")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before training.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

