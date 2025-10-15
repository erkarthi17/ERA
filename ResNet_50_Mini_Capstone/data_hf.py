"""
Data loading and preprocessing for ImageNet dataset using Hugging Face datasets.
This module provides a drop-in replacement for data.py that loads ImageNet from Hugging Face.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from typing import Tuple, Optional
import os
from PIL import Image


class ImageNetDataset(Dataset):
    """
    Custom Dataset wrapper for Hugging Face ImageNet dataset.
    """
    
    def __init__(self, dataset, transform=None, subset_size=None):
        """
        Args:
            dataset: Hugging Face dataset
            transform: Transform to apply to images
            subset_size: If specified, use only a subset of the data
        """
        self.dataset = dataset
        self.transform = transform
        
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


def get_train_transforms(config) -> transforms.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        config: Configuration object
    
    Returns:
        Composed transforms for training
    """
    return transforms.Compose([
        # Random resized crop
        transforms.RandomResizedCrop(
            size=config.image_size,
            scale=(config.min_crop_size, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        ),
        
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Color jitter
        transforms.ColorJitter(
            brightness=config.color_jitter,
            contrast=config.color_jitter,
            saturation=config.color_jitter,
            hue=0.0
        ),
        
        # Convert PIL to tensor
        transforms.ToTensor(),
        
        # Normalize
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(config) -> transforms.Compose:
    """
    Get validation data transforms.
    
    Args:
        config: Configuration object
    
    Returns:
        Composed transforms for validation
    """
    return transforms.Compose([
        # Resize to 256
        transforms.Resize(
            size=config.val_resize_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        ),
        
        # Center crop to 224
        transforms.CenterCrop(config.val_center_crop_size),
        
        # Convert PIL to tensor
        transforms.ToTensor(),
        
        # Normalize
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_imagenet_from_hf(train_subset_size=None, val_subset_size=None):
    """
    Load ImageNet dataset from Hugging Face.
    
    Args:
        train_subset_size: Size of training subset (None for full dataset)
        val_subset_size: Size of validation subset (None for full dataset)
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print("Loading ImageNet from Hugging Face...")
    
    try:
        # Load the dataset
        dataset = load_dataset("ILSVRC/imagenet-1k")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        
        print(f"Loaded ImageNet - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Apply subset if specified
        if train_subset_size is not None:
            print(f"Using training subset of size: {train_subset_size}")
        if val_subset_size is not None:
            print(f"Using validation subset of size: {val_subset_size}")
            
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"Error loading ImageNet from Hugging Face: {e}")
        print("Make sure you have access to the ImageNet dataset and are logged in to Hugging Face.")
        raise


def get_train_loader(config) -> DataLoader:
    """
    Get training data loader using Hugging Face dataset.
    
    Args:
        config: Configuration object
    
    Returns:
        Training DataLoader
    """
    # Get transforms
    train_transforms = get_train_transforms(config)
    
    # Load dataset from Hugging Face
    train_dataset_hf, _ = load_imagenet_from_hf(
        train_subset_size=getattr(config, 'train_subset_size', None)
    )
    
    # Create custom dataset
    train_dataset = ImageNetDataset(
        dataset=train_dataset_hf,
        transform=train_transforms,
        subset_size=getattr(config, 'train_subset_size', None)
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    return train_loader


def get_val_loader(config) -> DataLoader:
    """
    Get validation data loader using Hugging Face dataset.
    
    Args:
        config: Configuration object
    
    Returns:
        Validation DataLoader
    """
    # Get transforms
    val_transforms = get_val_transforms(config)
    
    # Load dataset from Hugging Face
    _, val_dataset_hf = load_imagenet_from_hf(
        val_subset_size=getattr(config, 'val_subset_size', None)
    )
    
    # Create custom dataset
    val_dataset = ImageNetDataset(
        dataset=val_dataset_hf,
        transform=val_transforms,
        subset_size=getattr(config, 'val_subset_size', None)
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


def get_data_loaders(config) -> Tuple[DataLoader, DataLoader]:
    """
    Get both training and validation data loaders using Hugging Face.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = get_train_loader(config)
    val_loader = get_val_loader(config)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading
    from config import Config
    
    config = Config()
    
    print("Creating data loaders with Hugging Face ImageNet...")
    
    try:
        train_loader, val_loader = get_data_loaders(config)
        
        print(f"\nTrain dataset size: {len(train_loader.dataset)}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test loading a batch
        print("\nLoading a training batch...")
        for images, labels in train_loader:
            print(f"Image shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Image dtype: {images.dtype}")
            print(f"Label dtype: {labels.dtype}")
            print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
            break
            
        print("\nHugging Face data loading test passed!")
        
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("Make sure you are logged in to Hugging Face and have access to ImageNet.")
