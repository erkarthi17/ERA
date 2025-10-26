import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import io
import s3fs
from tqdm import tqdm
import multiprocessing

# Define ImageNet mean and std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ImageNetS3Dataset(Dataset):
    """
    A PyTorch Dataset for loading ImageNet data directly from S3.
    """
    def __init__(self, s3_bucket, s3_prefix, transform=None, subset_size=None, s3_endpoint_url=None):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.transform = transform
        self.s3_endpoint_url = s3_endpoint_url

        if self.s3_endpoint_url:
            self.s3 = s3fs.S3FileSystem(client_kwargs={'endpoint_url': self.s3_endpoint_url})
        else:
            self.s3 = s3fs.S3FileSystem()

        # List all image files in the specified S3 prefix
        # We assume image files are .JPEG, .jpeg, .jpg etc.
        # This will be a list of full s3 paths, e.g., s3://bucket/prefix/n01440764/n01440764_10026.JPEG
        print(f"Listing files in s3://{s3_bucket}/{s3_prefix}...")
        try:
            # More efficient way to glob for multiple extensions
                        # More robust way to glob for multiple case-insensitive extensions
            s3_paths = []
            s3_paths.extend(self.s3.glob(f"{s3_bucket}/{s3_prefix}**/*.jpg"))
            s3_paths.extend(self.s3.glob(f"{s3_bucket}/{s3_prefix}**/*.jpeg"))
            s3_paths.extend(self.s3.glob(f"{s3_bucket}/{s3_prefix}**/*.JPG"))
            s3_paths.extend(self.s3.glob(f"{s3_bucket}/{s3_prefix}**/*.JPEG"))

            if not s3_paths:
                raise FileNotFoundError(f"No image files found in s3://{s3_bucket}/{s3_prefix}")
            
            # Filter to include only files, not directories or prefixes
            # For s3fs, glob typically returns files, but an explicit check can be added if issues arise
            self.s3_image_paths = sorted([f"s3://{path}" for path in s3_paths if self.s3.isfile(f"s3://{path}")])

            if not self.s3_image_paths:
                raise FileNotFoundError(f"No actual image files found after filtering in s3://{s3_bucket}/{s3_prefix}")

            # Extract class labels from path (e.g., n01440764)
            # This assumes the directory structure where the immediate parent folder of the image is its class
            self.classes = sorted(list(set([path.split('/')[-2] for path in self.s3_image_paths])))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            self.labels = [self.class_to_idx[path.split('/')[-2]] for path in self.s3_image_paths]

            # Apply subset if specified (for faster testing)
            if subset_size is not None and subset_size > 0:
                self.s3_image_paths = self.s3_image_paths[:subset_size]
                self.labels = self.labels[:subset_size]
                print(f"Using a subset of {len(self.s3_image_paths)} images from s3://{s3_bucket}/{s3_prefix}")

            print(f"Found {len(self.s3_image_paths)} images in s3://{s3_bucket}/{s3_prefix}")
            print(f"Number of classes: {len(self.classes)}")

        except Exception as e:
            print(f"Error listing S3 files: {e}")
            raise

    def __len__(self):
        return len(self.s3_image_paths)

    def __getitem__(self, idx):
        s3_path = self.s3_image_paths[idx]
        label = self.labels[idx]

        try:
            # Read image from S3
            with self.s3.open(s3_path, 'rb') as f:
                img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            if self.transform:
                img = self.transform(img)

            return img, label
        except Exception as e:
            print(f"Error loading image {s3_path}: {e}")
            # Optionally return a dummy image and label, or re-raise if it's a critical error
            # For robustness, you might want to skip this image and load the next,
            # but for simplicity, we'll raise.
            raise


def get_data_loaders(config):
    """
    Creates and returns the training and validation data loaders from S3.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = ImageNetS3Dataset(
        s3_bucket=config.s3_bucket,
        s3_prefix=config.s3_prefix_train,
        transform=train_transform,
        subset_size=getattr(config, 'train_subset_size', None)
    )

    val_dataset = ImageNetS3Dataset(
        s3_bucket=config.s3_bucket,
        s3_prefix=config.s3_prefix_val,
        transform=val_transform,
        subset_size=getattr(config, 'val_subset_size', None)
    )
    
    # Determine the multiprocessing start method for DataLoaders
    # This helps avoid 'fork-safe' issues with s3fs on some systems (e.g., Linux default 'fork')
    # 'forkserver' or 'spawn' are safer alternatives.
    mp_context = None
    if multiprocessing.get_start_method(allow_none=True) == 'fork':
        # Check if 'forkserver' is available, otherwise fall back to 'spawn'
        if 'forkserver' in multiprocessing.get_all_start_methods():
            mp_context = 'forkserver'
        elif 'spawn' in multiprocessing.get_all_start_methods():
            mp_context = 'spawn'
    
    print(f"Using multiprocessing context for DataLoader: {mp_context}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        multiprocessing_context=mp_context # Added multiprocessing_context
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        multiprocessing_context=mp_context # Added multiprocessing_context
    )
    return train_loader, val_loader