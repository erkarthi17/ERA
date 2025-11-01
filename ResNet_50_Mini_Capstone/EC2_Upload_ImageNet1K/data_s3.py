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
import sys
import hashlib

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _safe_cache_path(base_dir, s3_path):
    """Create a deterministic local cache path for an S3 object."""
    # Hash to avoid ultra-long filenames
    rel_key = s3_path.replace("s3://", "")
    hashed = hashlib.md5(rel_key.encode()).hexdigest()
    fname = os.path.basename(s3_path)
    return os.path.join(base_dir, f"{hashed}_{fname}")

class ImageNetS3Dataset(Dataset):
    """
    PyTorch Dataset that reads from S3 and transparently caches images locally.
    """
    def __init__(self, s3_bucket, s3_prefix, transform=None, subset_size=None,
                 s3_endpoint_url=None, cache_dir=".s3_cache", force_relist_s3=False,
                 max_retries=3):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.transform = transform
        self.s3_endpoint_url = s3_endpoint_url
        self.cache_dir = cache_dir
        self.force_relist_s3 = force_relist_s3
        self.max_retries = max_retries

        if self.s3_endpoint_url:
            self.s3 = s3fs.S3FileSystem(client_kwargs={'endpoint_url': self.s3_endpoint_url})
        else:
            self.s3 = s3fs.S3FileSystem()

        os.makedirs(self.cache_dir, exist_ok=True)

        # Cached file list
        cache_file = os.path.join(
            self.cache_dir,
            f"s3_file_list_{s3_bucket}_{s3_prefix.replace('/', '_')}.json"
        )

        if os.path.exists(cache_file) and not self.force_relist_s3:
            print(f"[{time.strftime('%H:%M:%S')}] Loading file list from {cache_file}")
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                self.s3_image_paths = data["s3_image_paths"]
            except Exception:
                print("Cache load failed — relisting S3.")
                self.s3_image_paths = self._list_s3_files()
        else:
            self.s3_image_paths = self._list_s3_files()
            with open(cache_file, "w") as f:
                json.dump({"s3_image_paths": self.s3_image_paths}, f)

        if not self.s3_image_paths:
            raise FileNotFoundError(f"No images found in s3://{s3_bucket}/{s3_prefix}")

        # Build class-to-index mapping
        self.classes = sorted({p.split("/")[-2] for p in self.s3_image_paths})
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.labels = [self.class_to_idx[p.split("/")[-2]] for p in self.s3_image_paths]

        if subset_size:
            self.s3_image_paths = self.s3_image_paths[:subset_size]
            self.labels = self.labels[:subset_size]
            print(f"[{time.strftime('%H:%M:%S')}] Using subset of {subset_size} images.")

        print(f"[{time.strftime('%H:%M:%S')}] Dataset ready: {len(self.s3_image_paths)} images, {len(self.classes)} classes.")

    def _list_s3_files(self):
        """List image files from S3 by extension."""
        print(f"[{time.strftime('%H:%M:%S')}] Listing files in s3://{self.s3_bucket}/{self.s3_prefix}...")
        start = time.time()
        paths = []
        for ext in ["jpg", "jpeg", "JPG", "JPEG"]:
            pattern = f"{self.s3_bucket}/{self.s3_prefix}**/*.{ext}"
            found = self.s3.glob(pattern)
            paths.extend([f"s3://{p}" for p in found])
        print(f"[{time.strftime('%H:%M:%S')}] Listed {len(paths)} files in {time.time()-start:.1f}s.")
        return sorted(paths)

    def __len__(self):
        return len(self.s3_image_paths)

    def _load_from_s3(self, s3_path):
        """Download with retries."""
        for attempt in range(self.max_retries):
            try:
                with self.s3.open(s3_path, "rb") as f:
                    return f.read()
            except Exception as e:
                print(f"Retry {attempt+1}/{self.max_retries} for {s3_path}: {e}")
                time.sleep(1 + attempt)
        raise IOError(f"Failed to load {s3_path} after {self.max_retries} retries")

    def __getitem__(self, idx):
        s3_path = self.s3_image_paths[idx]
        label = self.labels[idx]
        cache_path = _safe_cache_path(self.cache_dir, s3_path)

        # Try local cache first
        if os.path.exists(cache_path):
            try:
                img = Image.open(cache_path).convert("RGB")
            except Exception:
                os.remove(cache_path)
                img_bytes = self._load_from_s3(s3_path)
                with open(cache_path, "wb") as f:
                    f.write(img_bytes)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            img_bytes = self._load_from_s3(s3_path)
            with open(cache_path, "wb") as f:
                f.write(img_bytes)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label


def get_data_loaders(config):
    """Return train and validation loaders with caching."""
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
        config.s3_bucket, config.s3_prefix_train, transform=train_transform,
        subset_size=getattr(config, "train_subset_size", None),
        cache_dir=getattr(config, "cache_dir", ".s3_cache"),
        force_relist_s3=getattr(config, "force_relist_s3", False)
    )
    val_dataset = ImageNetS3Dataset(
        config.s3_bucket, config.s3_prefix_val, transform=val_transform,
        subset_size=getattr(config, "val_subset_size", None),
        cache_dir=getattr(config, "cache_dir", ".s3_cache"),
        force_relist_s3=getattr(config, "force_relist_s3", False)
    )

    mp_context = None
    if config.num_workers > 0:
        methods = multiprocessing.get_all_start_methods()
        mp_context = "forkserver" if "forkserver" in methods else "spawn"

    print(f"[{time.strftime('%H:%M:%S')}] Using multiprocessing context: {mp_context}")
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        multiprocessing_context=mp_context, timeout=getattr(config, "dataloader_timeout", 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        multiprocessing_context=mp_context, timeout=getattr(config, "dataloader_timeout", 0)
    )
    return train_loader, val_loader