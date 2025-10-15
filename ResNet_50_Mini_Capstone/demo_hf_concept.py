"""
Demo script showing Hugging Face datasets integration concept.
This uses a public dataset (CIFAR-10) to demonstrate the approach without requiring ImageNet access.
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from model import resnet50
import time

class HuggingFaceDataset(Dataset):
    """Generic wrapper for Hugging Face datasets."""
    
    def __init__(self, dataset, transform=None, subset_size=None):
        self.dataset = dataset
        self.transform = transform
        
        if subset_size is not None and subset_size < len(dataset):
            self.indices = list(range(subset_size))
        else:
            self.indices = list(range(len(dataset)))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        item = self.dataset[actual_idx]
        
        # Get image and label
        image = item['img']  # CIFAR-10 uses 'img' key
        label = item['label']
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def demo_hf_integration():
    """Demonstrate Hugging Face datasets integration."""
    print("="*60)
    print("DEMO: Hugging Face Datasets Integration")
    print("="*60)
    
    # Load CIFAR-10 from Hugging Face (public dataset)
    print("\n1. Loading CIFAR-10 from Hugging Face...")
    try:
        dataset = load_dataset("cifar10")
        train_dataset_hf = dataset["train"]
        test_dataset_hf = dataset["test"]
        print(f"   ✅ Successfully loaded CIFAR-10")
        print(f"   📊 Train: {len(train_dataset_hf)} samples")
        print(f"   📊 Test: {len(test_dataset_hf)} samples")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Create transforms
    print("\n2. Creating data transforms...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("   ✅ Transforms created")
    
    # Create datasets with small subsets
    print("\n3. Creating PyTorch datasets...")
    train_subset_size = 1000
    test_subset_size = 200
    
    train_dataset = HuggingFaceDataset(
        dataset=train_dataset_hf,
        transform=transform,
        subset_size=train_subset_size
    )
    
    test_dataset = HuggingFaceDataset(
        dataset=test_dataset_hf,
        transform=transform,
        subset_size=test_subset_size
    )
    
    print(f"   ✅ Created datasets with subsets:")
    print(f"   📊 Train subset: {len(train_dataset)} samples")
    print(f"   📊 Test subset: {len(test_dataset)} samples")
    
    # Create data loaders
    print("\n4. Creating data loaders...")
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"   ✅ Data loaders created (batch_size={batch_size})")
    
    # Test data loading
    print("\n5. Testing data loading...")
    start_time = time.time()
    
    for i, (images, labels) in enumerate(train_loader):
        print(f"   📦 Batch {i+1}: images={images.shape}, labels={labels.shape}")
        print(f"   📊 Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"   🏷️  Label range: [{labels.min()}, {labels.max()}]")
        if i >= 2:  # Test first 3 batches
            break
    
    load_time = time.time() - start_time
    print(f"   ✅ Data loading successful (took {load_time:.2f}s)")
    
    # Test model forward pass
    print("\n6. Testing model forward pass...")
    try:
        # Create ResNet-50 model (modify for CIFAR-10: 10 classes)
        model = resnet50(num_classes=10)  # CIFAR-10 has 10 classes
        model.eval()
        
        # Test with a batch
        for images, labels in train_loader:
            with torch.no_grad():
                start_time = time.time()
                outputs = model(images)
                forward_time = time.time() - start_time
            
            print(f"   🧠 Model forward pass successful:")
            print(f"   📊 Input shape: {images.shape}")
            print(f"   📊 Output shape: {outputs.shape}")
            print(f"   ⏱️  Forward time: {forward_time:.3f}s")
            print(f"   🔢 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            break
        
        print("   ✅ Model integration successful")
        
    except Exception as e:
        print(f"   ❌ Model error: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nThis demonstrates how to:")
    print("✅ Load datasets from Hugging Face")
    print("✅ Create PyTorch datasets and data loaders")
    print("✅ Apply transforms and preprocessing")
    print("✅ Integrate with PyTorch models")
    print("✅ Use subsets for testing")
    
    print("\n🚀 For ImageNet training:")
    print("1. Get Hugging Face account with ImageNet access")
    print("2. Run: huggingface-cli login")
    print("3. Use the same approach with 'imagenet-1k' dataset")
    print("4. Run: python train_hf.py --train-subset 1000 --epochs 1")
    
    return True

def show_code_comparison():
    """Show the code structure comparison."""
    print("\n" + "="*60)
    print("📝 CODE STRUCTURE COMPARISON")
    print("="*60)
    
    print("\n🔹 Original approach (local files):")
    print("""
    # data.py - Original
    train_dataset = torchvision.datasets.ImageFolder(
        root=config.train_path,  # Local directory
        transform=train_transforms
    )
    """)
    
    print("\n🔹 Hugging Face approach:")
    print("""
    # data_hf.py - Hugging Face
    dataset = load_dataset("imagenet-1k")  # From Hugging Face
    train_dataset_hf = dataset["train"]
    
    train_dataset = ImageNetDataset(
        dataset=train_dataset_hf,
        transform=train_transforms,
        subset_size=config.train_subset_size  # Easy subsetting
    )
    """)
    
    print("\n💡 Key advantages:")
    print("   • No local file management")
    print("   • Automatic download and caching")
    print("   • Built-in subsetting")
    print("   • Version control and updates")
    print("   • Cloud-ready")

if __name__ == "__main__":
    success = demo_hf_integration()
    
    if success:
        show_code_comparison()
        
        print("\n" + "="*60)
        print("🎯 READY FOR IMAGENET TRAINING!")
        print("="*60)
        print("\nTo proceed with ImageNet:")
        print("1. Get Hugging Face account")
        print("2. Request ImageNet access")
        print("3. Run: huggingface-cli login")
        print("4. Run: python train_hf.py --train-subset 1000 --epochs 1")
    else:
        print("\n❌ Demo failed. Check your setup.")
