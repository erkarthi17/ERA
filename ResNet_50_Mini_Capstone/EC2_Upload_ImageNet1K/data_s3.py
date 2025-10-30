import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import io
import s3fs
from tqdm import tqdm
import multiprocessing
import json
import time
import sys # Added to specify tqdm output stream

# Define ImageNet mean and std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ImageNetS3Dataset(Dataset):
    """
    A PyTorch Dataset for loading ImageNet data directly from S3.
    """
    def __init__(self, s3_bucket, s3_prefix, transform=None, subset_size=None, 
                 s3_endpoint_url=None, cache_dir=None, force_relist_s3=False):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.transform = transform
        self.s3_endpoint_url = s3_endpoint_url
        self.cache_dir = cache_dir
        self.force_relist_s3 = force_relist_s3

        if self.s3_endpoint_url:
            self.s3 = s3fs.S3FileSystem(client_kwargs={'endpoint_url': self.s3_endpoint_url})
        else:
            self.s3 = s3fs.S3FileSystem()

        self.s3_image_paths = []
        self.classes = []
        self.class_to_idx = {}
        self.labels = []

        # Construct cache file path
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Create a unique cache filename based on bucket and prefix
            cache_filename = f"s3_file_list_{s3_bucket}_{s3_prefix.replace('/', '_')}.json"
            self.cache_filepath = os.path.join(self.cache_dir, cache_filename)
        else:
            self.cache_filepath = None

        # Try to load from cache first
        if self.cache_filepath and os.path.exists(self.cache_filepath) and not self.force_relist_s3:
            print(f"[{time.strftime('%H:%M:%S')}] Loading S3 file list from cache: {self.cache_filepath}...")
            try:
                with open(self.cache_filepath, 'r') as f:
                    cached_data = json.load(f)
                self.s3_image_paths = cached_data['s3_image_paths']
                print(f"[{time.strftime('%H:%M:%S')}] Successfully loaded {len(self.s3_image_paths)} paths from cache.")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Error loading from cache: {e}. Re-listing from S3.")
                self._list_s3_files() # Fallback to S3 listing
        else:
            # If no cache or forced relist, list from S3
            self._list_s3_files()

        # After paths are loaded (either from cache or S3), derive labels
        if not self.s3_image_paths:
            raise FileNotFoundError(f"No image files found in s3://{s3_bucket}/{s3_prefix}")
        
        # Extract class labels from path (e.g., n01440764)
        # This assumes the directory structure where the immediate parent folder of the image is its class
        self.classes = sorted(list(set([path.split('/')[-2] for path in self.s3_image_paths])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.labels = [self.class_to_idx[path.split('/')[-2]] for path in self.s3_image_paths]

        # Apply subset if specified (for faster testing)
        if subset_size is not None and subset_size > 0:
            self.s3_image_paths = self.s3_image_paths[:subset_size]
            self.labels = self.labels[:subset_size]
            print(f"[{time.strftime('%H:%M:%S')}] Using a subset of {len(self.s3_image_paths)} images from s3://{s3_bucket}/{s3_prefix}")

        print(f"[{time.strftime('%H:%M:%S')}] Found {len(self.s3_image_paths)} images in s3://{s3_bucket}/{s3_prefix}")
        print(f"[{time.strftime('%H:%M:%S')}] Number of classes: {len(self.classes)}")

    def _list_s3_files(self):
        """Helper method to list files from S3 and optionally save to cache."""
        print(f"[{time.strftime('%H:%M:%S')}] Listing files directly from s3://{self.s3_bucket}/{self.s3_prefix}...")
        try:
            start_time = time.time()
            s3_paths_collector = []
            
            extensions = ["jpg", "jpeg", "JPG", "JPEG"]
            
            # Use tqdm to show progress through the extensions
            with tqdm(extensions, desc="Globbing S3 by extension", dynamic_ncols=True, file=sys.stdout) as pbar:
                for ext in pbar:
                    current_glob_pattern = f"{self.s3_bucket}/{self.s3_prefix}**/*.{ext}"
                    pbar.set_postfix_str(f"Globbing for .{ext}...")
                    
                    print(f"[{time.strftime('%H:%M:%S')}] Starting glob for: {current_glob_pattern}")
                    
                    found_paths_for_ext = self.s3.glob(current_glob_pattern)
                    s3_paths_collector.extend([f"s3://{path}" for path in found_paths_for_ext])
                    
                    pbar.set_postfix_str(f"Found {len(found_paths_for_ext)} .{ext} files. Total: {len(s3_paths_collector)}")
                    print(f"[{time.strftime('%H:%M:%S')}] Finished glob for .{ext}. Found {len(found_paths_for_ext)} files. Cumulative total: {len(s3_paths_collector)}")

            if not s3_paths_collector:
                raise FileNotFoundError(f"No image files found in s3://{self.s3_bucket}/{self.s3_prefix}")
            
            self.s3_image_paths = sorted(s3_paths_collector)

            if not self.s3_image_paths:
                raise FileNotFoundError(f"No actual image files found after filtering in s3://{self.s3_bucket}/{self.s3_prefix}")
            
            end_time = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] S3 listing completed in {end_time - start_time:.2f} seconds. Total unique images found: {len(self.s3_image_paths)}.")

            # Save to cache if cache_filepath is defined
            if self.cache_filepath:
                with open(self.cache_filepath, 'w') as f:
                    json.dump({'s3_image_paths': self.s3_image_paths, 'timestamp': time.time()}, f)
                print(f"[{time.strftime('%H:%M:%S')}] Saved S3 file list to cache: {self.cache_filepath}")

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error listing S3 files: {e}")
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
            print(f"[{time.strftime('%H:%M:%S')}] Error loading image {s3_path}: {e}")
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
        subset_size=getattr(config, 'train_subset_size', None),
        cache_dir=getattr(config, 'cache_dir', None),
        force_relist_s3=getattr(config, 'force_relist_s3', False)
    )

    val_dataset = ImageNetS3Dataset(
        s3_bucket=config.s3_bucket,
        s3_prefix=config.s3_prefix_val,
        transform=val_transform,
        subset_size=getattr(config, 'val_subset_size', None),
        cache_dir=getattr(config, 'cache_dir', None),
        force_relist_s3=getattr(config, 'force_relist_s3', False)
    )
    
    # Determine the multiprocessing start method for DataLoaders
    mp_context = None
    if config.num_workers > 0:
        if 'forkserver' in multiprocessing.get_all_start_methods():
            mp_context = 'forkserver'
        elif 'spawn' in multiprocessing.get_all_start_methods():
            mp_context = 'spawn'
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Warning: Neither 'forkserver' nor 'spawn' available. DataLoader might fail.")
    
    print(f"[{time.strftime('%H:%M:%S')}] Using multiprocessing context for DataLoader: {mp_context}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        multiprocessing_context=mp_context
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        multiprocessing_context=mp_context
    )
    return train_loader, val_loader