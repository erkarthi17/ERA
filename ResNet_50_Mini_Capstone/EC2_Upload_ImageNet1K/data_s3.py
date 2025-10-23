import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional
import os
from PIL import Image
import s3fs
import io
import pandas as pd # Assuming you have a CSV or similar for labels, or infer from path

# Assuming config is in the same directory
from .config import Config

class ImageNetS3Dataset(Dataset):
    """
    Custom Dataset for ImageNet loaded directly from S3.
    Assumes a structure like s3://bucket/prefix/train/n01440764/n01440764_10026.JPEG
    and infers labels from parent directory names.
    """
    def __init__(self, s3_bucket: str, s3_prefix: str, transform=None, subset_size: Optional[int] = None):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.transform = transform
        self.s3 = s3fs.S3FileSystem()
        self.samples = [] # List of (s3_path, label)
        self.class_to_idx = {}
        self.idx_to_class = {}

        print(f"Listing files in s3://{s3_bucket}/{s3_prefix}...")
        s3_paths = self.s3.glob(f"{s3_bucket}/{s3_prefix}**/*.jpg")

        # Create a mapping of class names to indices
        class_names = sorted(list(set([path.split('/')[-2] for path in s3_paths])))
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
        
        for s3_path in s3_paths:
            class_name = s3_path.split('/')[-2]
            label = self.class_to_idx[class_name]
            self.samples.append((s3_path, label))

        if subset_size is not None and subset_size < len(self.samples):
            self.samples = self.samples[:subset_size]

        print(f"Found {len(self.samples)} images in s3://{s3_bucket}/{s3_prefix}")
        print(f"Number of classes: {len(self.class_to_idx)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s3_path, label = self.samples[idx]

        with self.s3.open(s3_path, 'rb') as f:
            image_bytes = f.read()
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_train_transforms(config: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=config.image_size,
            scale=(config.min_crop_size, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=config.color_jitter,
            contrast=config.color_jitter,
            saturation=config.color_jitter,
            hue=0.0
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms(config: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(
            size=config.val_resize_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        ),
        transforms.CenterCrop(config.val_center_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Get both training and validation data loaders from S3.
    """
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)

    train_dataset = ImageNetS3Dataset(
        s3_bucket=config.s3_bucket,
        s3_prefix=config.s3_prefix_train,
        transform=train_transforms,
        subset_size=getattr(config, 'train_subset_size', None)
    )
    val_dataset = ImageNetS3Dataset(
        s3_bucket=config.s3_bucket,
        s3_prefix=config.s3_prefix_val,
        transform=val_transforms,
        subset_size=getattr(config, 'val_subset_size', None)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test data loading
    config = Config()
    config.s3_bucket = "imagenet-dataset-karthick-kannan"
    config.s3_prefix_train = "imagenet-1k/train/"
    config.s3_prefix_val = "imagenet-1k/val/"

    print("Creating data loaders from S3...")

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
            
        print("\nS3 data loading test passed!")
        
    except Exception as e:
        print(f"\nError loading data from S3: {e}")
        print("Make sure your AWS credentials are configured correctly and the S3 bucket/prefix exist.")