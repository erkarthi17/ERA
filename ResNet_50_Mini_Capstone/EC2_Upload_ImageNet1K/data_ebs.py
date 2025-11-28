import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(config):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Determine interpolation mode based on config
    # Use config.interpolation, default to transforms.InterpolationMode.BILINEAR
    interpolation_mode = getattr(config, 'interpolation', 'bilinear').lower()
    if interpolation_mode == 'bicubic':
        interpolation = transforms.InterpolationMode.BICUBIC
    else:
        interpolation = transforms.InterpolationMode.BILINEAR

    train_transform_list.extend([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(
        os.path.expanduser(config.train_path),
        transforms.Compose(train_transform_list)
    )

    val_dataset = datasets.ImageFolder(
        os.path.expanduser(config.val_path),
        transforms.Compose([
            transforms.Resize(config.val_resize_size, interpolation=interpolation),
            transforms.CenterCrop(config.val_center_crop_size),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=getattr(config, 'persistent_workers', False), # Use getattr for robustness
        prefetch_factor=getattr(config, 'prefetch_factor', 2), # Add prefetch_factor
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size, # Use val_batch_size from config
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=getattr(config, 'persistent_workers', False), # Use getattr for robustness
        prefetch_factor=getattr(config, 'prefetch_factor', 2), # Add prefetch_factor
    )

    return train_loader, val_loader