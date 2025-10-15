"""
Alternative data loading using other Hugging Face datasets.
Use this if you don't have access to ImageNet-1k.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from typing import Tuple, Optional
import os
from PIL import Image


class AlternativeDataset(Dataset):
    """
    Custom Dataset wrapper for alternative Hugging Face datasets.
    """
    
    def __init__(self, dataset, transform=None, subset_size=None, num_classes=1000):
        """
        Args:
            dataset: Hugging Face dataset
            transform: Transform to apply to images
            subset_size: If specified, use only a subset of the data
            num_classes: Number of classes (for model compatibility)
        """
        self.dataset = dataset
        self.transform = transform
        self.num_classes = num_classes
        
        # Limit dataset size if specified
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
        image = item['image']
        label = item['label']
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_available_datasets():
    """Get list of available alternative datasets."""
    return {
        'cifar10': {
            'name': 'CIFAR-10',
            'classes': 10,
            'description': '10-class image classification (airplane, car, etc.)'
        },
        'food101': {
            'name': 'Food-101',
            'classes': 101,
            'description': '101-class food classification'
        },
        'oxford_flowers102': {
            'name': 'Oxford Flowers 102',
            'classes': 102,
            'description': '102-class flower classification'
        },
        'stanford_cars': {
            'name': 'Stanford Cars',
            'classes': 196,
            'description': '196-class car classification'
        }
    }


def load_alternative_dataset(dataset_name="cifar10", train_subset_size=None, val_subset_size=None):
    """
    Load alternative dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset to load
        train_subset_size: Size of training subset (None for full dataset)
        val_subset_size: Size of validation subset (None for full dataset)
    
    Returns:
        Tuple of (train_dataset, val_dataset, num_classes)
    """
    available_datasets = get_available_datasets()
    
    if dataset_name not in available_datasets:
        print(f"❌ Dataset '{dataset_name}' not available.")
        print(f"Available datasets: {list(available_datasets.keys())}")
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_info = available_datasets[dataset_name]
    num_classes = dataset_info['classes']
    
    print(f"Loading {dataset_info['name']} from Hugging Face...")
    print(f"Description: {dataset_info['description']}")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)
        
        # Handle different dataset structures
        if 'train' in dataset:
            train_dataset = dataset["train"]
        elif 'train' in dataset:
            train_dataset = dataset["train"]
        else:
            # Some datasets might have different split names
            train_dataset = dataset[list(dataset.keys())[0]]
        
        if 'validation' in dataset:
            val_dataset = dataset["validation"]
        elif 'test' in dataset:
            val_dataset = dataset["test"]
        else:
            # Create validation split from training data
            val_dataset = train_dataset.select(range(min(1000, len(train_dataset) // 5)))
        
        print(f"Loaded {dataset_info['name']} - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"Number of classes: {num_classes}")
        
        # Apply subset if specified
        if train_subset_size is not None:
            print(f"Using training subset of size: {train_subset_size}")
        if val_subset_size is not None:
            print(f"Using validation subset of size: {val_subset_size}")
            
        return train_dataset, val_dataset, num_classes
        
    except Exception as e:
        print(f"Error loading {dataset_info['name']}: {e}")
        raise


def get_train_transforms(config, image_size=224) -> transforms.Compose:
    """
    Get training data augmentation transforms.
    """
    return transforms.Compose([
        # Resize to target size
        transforms.Resize((image_size, image_size)),
        
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Color jitter
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        
        # Convert PIL to tensor
        transforms.ToTensor(),
        
        # Normalize (ImageNet stats)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(config, image_size=224) -> transforms.Compose:
    """
    Get validation data transforms.
    """
    return transforms.Compose([
        # Resize to target size
        transforms.Resize((image_size, image_size)),
        
        # Convert PIL to tensor
        transforms.ToTensor(),
        
        # Normalize (ImageNet stats)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_train_loader(config, dataset_name="cifar10") -> DataLoader:
    """
    Get training data loader using alternative dataset.
    """
    # Get transforms
    train_transforms = get_train_transforms(config)
    
    # Load dataset from Hugging Face
    train_dataset_hf, _, num_classes = load_alternative_dataset(
        dataset_name=dataset_name,
        train_subset_size=getattr(config, 'train_subset_size', None)
    )
    
    # Update config with actual number of classes
    config.num_classes = num_classes
    
    # Create custom dataset
    train_dataset = AlternativeDataset(
        dataset=train_dataset_hf,
        transform=train_transforms,
        subset_size=getattr(config, 'train_subset_size', None),
        num_classes=num_classes
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    return train_loader


def get_val_loader(config, dataset_name="cifar10") -> DataLoader:
    """
    Get validation data loader using alternative dataset.
    """
    # Get transforms
    val_transforms = get_val_transforms(config)
    
    # Load dataset from Hugging Face
    _, val_dataset_hf, num_classes = load_alternative_dataset(
        dataset_name=dataset_name,
        val_subset_size=getattr(config, 'val_subset_size', None)
    )
    
    # Create custom dataset
    val_dataset = AlternativeDataset(
        dataset=val_dataset_hf,
        transform=val_transforms,
        subset_size=getattr(config, 'val_subset_size', None),
        num_classes=num_classes
    )
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    return val_loader


def get_data_loaders(config, dataset_name="cifar10") -> Tuple[DataLoader, DataLoader]:
    """
    Get both training and validation data loaders using alternative dataset.
    """
    train_loader = get_train_loader(config, dataset_name)
    val_loader = get_val_loader(config, dataset_name)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading with CIFAR-10
    from config import Config
    
    config = Config()
    config.train_subset_size = 100
    config.val_subset_size = 50
    config.batch_size = 16
    config.num_workers = 0
    
    print("Available datasets:")
    for name, info in get_available_datasets().items():
        print(f"  {name}: {info['name']} ({info['classes']} classes)")
    
    print(f"\nTesting with CIFAR-10...")
    
    try:
        train_loader, val_loader = get_data_loaders(config, "cifar10")
        
        print(f"\nTrain dataset size: {len(train_loader.dataset)}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Number of classes: {config.num_classes}")
        
        # Test loading a batch
        print("\nLoading a training batch...")
        for images, labels in train_loader:
            print(f"Image shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Label range: [{labels.min()}, {labels.max()}]")
            break
            
        print("\nAlternative dataset loading test passed!")
        
    except Exception as e:
        print(f"\nError loading data: {e}")
        import traceback
        traceback.print_exc()
