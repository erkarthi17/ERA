"""
Data loading and preprocessing for ImageNet dataset.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from typing import Tuple, Optional
import os


def get_train_transforms(config) -> transforms.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        config: Configuration object
    
    Returns:
        Composed transforms for training
    """
    return transforms.Compose([
        # Convert PIL to tensor first
        transforms.ToTensor(),
        
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
        # Convert PIL to tensor
        transforms.ToTensor(),
        
        # Resize to 256
        transforms.Resize(
            size=config.val_resize_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        ),
        
        # Center crop to 224
        transforms.CenterCrop(config.val_center_crop_size),
        
        # Normalize
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_train_loader(config) -> DataLoader:
    """
    Get training data loader.
    
    Args:
        config: Configuration object
    
    Returns:
        Training DataLoader
    """
    # Get transforms
    train_transforms = get_train_transforms(config)
    
    # Create dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root=config.train_path,
        transform=train_transforms
    )
    
    # Create sampler (for distributed training)
    if config.distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.local_rank,
            shuffle=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    return train_loader


def get_val_loader(config) -> DataLoader:
    """
    Get validation data loader.
    
    Args:
        config: Configuration object
    
    Returns:
        Validation DataLoader
    """
    # Get transforms
    val_transforms = get_val_transforms(config)
    
    # Create dataset
    val_dataset = torchvision.datasets.ImageFolder(
        root=config.val_path,
        transform=val_transforms
    )
    
    # Create sampler (for distributed training)
    if config.distributed:
        sampler = DistributedSampler(
            val_dataset,
            num_replicas=config.world_size,
            rank=config.local_rank,
            shuffle=False
        )
    else:
        sampler = None
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    return val_loader


def get_data_loaders(config) -> Tuple[DataLoader, DataLoader]:
    """
    Get both training and validation data loaders.
    
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
    config.data_root = "/path/to/imagenet"  # Update this
    
    print("Creating data loaders...")
    print(f"Train path: {config.train_path}")
    print(f"Val path: {config.val_path}")
    
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
            
        print("\nData loading test passed!")
        
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("Please update the data_root in config.py")

