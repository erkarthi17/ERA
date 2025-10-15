"""
Quick test training script for EC2.
Tests the training pipeline with minimal epochs and reduced batch size.
Supports multiple dataset options for local testing.
"""

import sys
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import tempfile
import shutil

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Clear any cached imports that might have old config
if 'config' in sys.modules:
    del sys.modules['config']
if 'data' in sys.modules:
    del sys.modules['data']

def create_cifar10_dataset(data_root):
    """
    Create CIFAR-10 dataset in ImageNet-like structure for testing.
    This creates actual image files that the ImageFolder loader can use.
    
    Args:
        data_root: Root directory where to create the dataset
    
    Returns:
        Path to the created dataset
    """
    print("Creating CIFAR-10 dataset for testing...")
    
    # Download CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root=tempfile.gettempdir(), 
        train=True, 
        download=True
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=tempfile.gettempdir(), 
        train=False, 
        download=True
    )
    
    # Create directory structure
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Map CIFAR-10 classes to dummy ImageNet-like numerical IDs
    # This is crucial for ImageFolder to recognize the class folders
    cifar10_class_names = train_dataset.classes
    imagenet_like_class_ids = [f"n{i:08d}" for i in range(len(cifar10_class_names))]
    
    # Create class directories
    for class_id in imagenet_like_class_ids:
        train_class_dir = os.path.join(train_dir, class_id)
        val_class_dir = os.path.join(val_dir, class_id)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
    
    # CIFAR-10 already returns PIL Images, so we can save them directly
    from PIL import Image
    
    print("Converting CIFAR-10 train data...")
    for idx, (image, label) in enumerate(train_dataset):
        if idx >= 1000:  # Limit to 1000 images per class for faster testing
            break
            
        # Save to appropriate class directory using ImageNet-like ID
        class_id = imagenet_like_class_ids[label]
        class_dir = os.path.join(train_dir, class_id)
        image_path = os.path.join(class_dir, f"train_{idx:06d}.png")
        image.save(image_path)
    
    print("Converting CIFAR-10 validation data...")
    for idx, (image, label) in enumerate(val_dataset):
        if idx >= 200:  # Limit to 200 images per class for faster testing
            break
            
        # Save to appropriate class directory using ImageNet-like ID
        class_id = imagenet_like_class_ids[label]
        class_dir = os.path.join(val_dir, class_id)
        image_path = os.path.join(class_dir, f"val_{idx:06d}.png")
        image.save(image_path)
    
    print(f"CIFAR-10 dataset ready at: {data_root}")
    print(f"Train samples: ~{len(train_dataset)} (limited to 1000 per class)")
    print(f"Val samples: ~{len(val_dataset)} (limited to 200 per class)")
    
    return data_root


def create_mini_imagenet_structure(data_root):
    """
    Create a minimal ImageNet-like structure using a small subset of images.
    This creates a structure that the existing data loader can use.
    
    Args:
        data_root: Root directory where to create the dataset
    
    Returns:
        Path to the created dataset
    """
    print("Creating mini ImageNet structure for testing...")
    
    # Create directory structure
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create 10 fake class directories (like ImageNet classes)
    fake_classes = [
        'n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475',
        'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878'
    ]
    
    for class_name in fake_classes:
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Create empty files as placeholders (the data loader will handle missing images gracefully)
        print(f"Created directory: {class_name}")
    
    print(f"Mini ImageNet structure created at: {data_root}")
    print("Note: This creates the directory structure but no actual images.")
    print("For real testing, you would need to populate these directories with actual images.")
    
    return data_root


def test_training_quick():
    """
    Run a quick test training with minimal settings using CIFAR-10.
    This is a legacy function - use the main function with --dataset cifar10 instead.
    """
    print("This function is deprecated. Use:")
    print("python test_training.py --dataset cifar10 --epochs 2 --batch-size 32")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick test training with different dataset options')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs (default: 2)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--data-root', type=str, help='Path to ImageNet data (optional if using --dataset)')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'mini-imagenet', 'custom'], 
                       default='cifar10', help='Dataset to use for testing (default: cifar10)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("QUICK TRAINING TEST")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Dataset: {args.dataset}")
    
    # Determine data root based on dataset choice
    if args.dataset == 'cifar10':
        # Use CIFAR-10 for testing
        data_root = os.path.join(tempfile.gettempdir(), 'test_imagenet_cifar10')
        create_cifar10_dataset(data_root)
        print(f"Using CIFAR-10 dataset at: {data_root}")
        
    elif args.dataset == 'mini-imagenet':
        # Create mini ImageNet structure
        data_root = os.path.join(tempfile.gettempdir(), 'test_imagenet_mini')
        create_mini_imagenet_structure(data_root)
        print(f"Using mini ImageNet structure at: {data_root}")
        
    elif args.dataset == 'custom':
        # Use provided data root
        if not args.data_root:
            # Use the local data/imagenet folder you created
            data_root = os.path.join(os.getcwd(), 'data', 'imagenet')
            print(f"Using local data/imagenet folder at: {data_root}")
        else:
            data_root = args.data_root
            print(f"Using custom dataset at: {data_root}")
    
    print("="*80)
    print()
    
    # Import and run main with test arguments
    import sys
    original_argv = sys.argv.copy()
    
    # Build command line arguments
    sys.argv = [
        'test_training.py',
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--data-root', data_root
    ]
    
    try:
        # Simple approach: just run with command line arguments
        # The train.py script will handle the config properly
        from train import main
        main()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(f"Data root used: {data_root}")
        print(f"Train path: {os.path.join(data_root, 'train')}")
        print(f"Val path: {os.path.join(data_root, 'val')}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have enough disk space")
        print("2. Check if the data path exists and has proper permissions")
        print("3. Try using --dataset cifar10 for a quick test")
        print("4. Note: CIFAR-10 has 10 classes, but the model expects 1000 ImageNet classes")
        raise
    finally:
        sys.argv = original_argv