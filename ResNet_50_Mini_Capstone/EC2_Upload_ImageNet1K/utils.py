"""
Utility functions for training, logging, and metrics.
"""

import torch
import torch.nn as nn
import time
import os
from typing import Dict, Optional
from pathlib import Path
import logging
from datetime import datetime
import sys
import boto3 # Added boto3 import for S3 interaction
from urllib.parse import urlparse # Added for parsing S3 URLs

# Initialize S3 client once, outside functions, for efficiency
s3_client = boto3.client('s3')

def setup_logging(log_dir: str, name: str = "train") -> logging.Logger:
    """
    Setup logging to file and console.
    
    Args:
        log_dir: Directory to save log files
        name: Name of the log file
    
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        # Improved formatting for console output
        # Example: Loss 0.1234 (0.1111)
        # Acc@1 80.12 (79.50)
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"


class ProgressMeter:
    """Display progress during training."""
    
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.num_batches = num_batches # Ensure num_batches is stored

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        
        # Use carriage return to update the same line in the terminal
        sys.stdout.write('\r' + '  '.join(entries))
        sys.stdout.flush()

        # At the very last batch, print a final newline to ensure the prompt appears on a new line.
        # This avoids double newlines when logger.info also prints.
        if batch == self.num_batches - 1:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Summary:
    """
    Computes and stores the average and current value.
    This is for summarizing at the end of an epoch/validation.
    """
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:{self.fmt}}"


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1, 5)):
    """
    Computes the accuracy over the k top predictions.
    
    Args:
        output: Model predictions (batch_size, num_classes)
        target: Ground truth labels (batch_size,)
        topk: Tuple of top-k accuracies to compute
    
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


def save_checkpoint(
    state: Dict,
    is_best: bool,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth',
    s3_bucket: Optional[str] = None, # Added S3 bucket parameter
    s3_prefix: Optional[str] = None   # Added S3 prefix parameter
):
    """
    Save training checkpoint locally and optionally upload to S3.
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints locally
        filename: Name of the checkpoint file
        s3_bucket: S3 bucket to upload to (optional)
        s3_prefix: S3 prefix (folder) within the bucket to upload to (optional)
    """
    # Get checkpoint name safely, whether config is dict or Config object
    cfg = state.get('config', None)
    if isinstance(cfg, dict):
        ckpt_name = cfg.get('checkpoint_name', 'default')
    elif hasattr(cfg, 'checkpoint_name'):
        ckpt_name = cfg.checkpoint_name
    else:
        ckpt_name = 'default'
    logger = logging.getLogger(ckpt_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save regular checkpoint locally
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Saved local checkpoint to {filepath}")

    # Upload regular checkpoint to S3
    if s3_bucket and s3_prefix:
        s3_object_key = os.path.join(s3_prefix, filename)
        try:
            s3_client.upload_file(filepath, s3_bucket, s3_object_key)
            logger.info(f"Uploaded checkpoint to s3://{s3_bucket}/{s3_object_key}")
        except Exception as e:
            logger.error(f"Error uploading checkpoint {filename} to S3: {e}")

    # Save and upload best model if applicable
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth') # Changed from best_model.pth for consistency
        torch.save(state, best_filepath)
        logger.info(f"Saved local best model to {best_filepath}")

        if s3_bucket and s3_prefix:
            s3_best_object_key = os.path.join(s3_prefix, 'model_best.pth')
            try:
                s3_client.upload_file(best_filepath, s3_bucket, s3_best_object_key)
                logger.info(f"Uploaded best model to s3://{s3_bucket}/{s3_best_object_key}")
            except Exception as e:
                logger.error(f"Error uploading best model to S3: {e}")


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer=None, scheduler=None, logger: Optional[logging.Logger] = None):
    """
    Load training checkpoint from a local path or an S3 path.
    Downloads from S3 to /tmp/ if an S3 path is provided.
    
    Args:
        checkpoint_path: Path to checkpoint file (local or S3 URL)
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Learning rate scheduler to load state into (optional)
        logger: Logger instance for output (optional)
    
    Returns:
        Dictionary containing loaded state
    """
    current_logger = logger if logger else logging.getLogger(__name__) # Use provided logger or create a default one
    local_path_to_load = checkpoint_path
    
    # Check if checkpoint_path is an S3 URL
    if checkpoint_path.startswith('s3://'):
        s3_url_parsed = urlparse(checkpoint_path)
        s3_bucket = s3_url_parsed.netloc
        s3_object_key = s3_url_parsed.path.lstrip('/')
        
        # Download from S3 to a temporary local file
        os.makedirs('/tmp', exist_ok=True) 
        local_path_to_load = os.path.join('/tmp', os.path.basename(s3_object_key))
        
        current_logger.info(f"Attempting to download checkpoint from s3://{s3_bucket}/{s3_object_key} to {local_path_to_load}")
        try:
            # Note: Assumes s3_client is initialized globally in this file
            s3_client.download_file(s3_bucket, s3_object_key, local_path_to_load)
            current_logger.info("Download successful. Loading checkpoint...")
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise FileNotFoundError(f"Checkpoint not found on S3: {checkpoint_path}")
            else:
                raise RuntimeError(f"Error downloading checkpoint from S3: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error downloading checkpoint: {e}")

    # Load the checkpoint from the local path (either original or downloaded)
    if not os.path.isfile(local_path_to_load):
        raise FileNotFoundError(f"Checkpoint file not found: {local_path_to_load}")

    current_logger.info(f"Loading checkpoint from {local_path_to_load}")
    checkpoint = torch.load(local_path_to_load, map_location='cpu', weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Conditionally load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_logger.info("Loaded optimizer state from checkpoint.")
    elif optimizer is not None:
        current_logger.warning("Optimizer state not found in checkpoint. Skipping.")
    
    # Conditionally load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_logger.info("Loaded scheduler state from checkpoint.")
    elif scheduler is not None:
        current_logger.warning("Scheduler state not found in checkpoint. Skipping.")
    
    current_logger.info(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}"
        f" batch {checkpoint.get('batch_idx', 'N/A')}"
        f" with best_acc1: {checkpoint.get('best_acc1', 0):.2f}%"
    )
    
    return checkpoint


def get_latest_s3_checkpoint(s3_bucket: str, s3_prefix: str, logger: Optional[logging.Logger] = None, checkpoint_filename_pattern: str = 'checkpoint_s3_epoch_*.pth') -> Optional[str]:
    """
    Finds the latest checkpoint in a given S3 bucket and prefix based on modification time.
    Returns the full S3 path of the latest checkpoint, or None if no checkpoints are found.
    """
    current_logger = logger if logger else logging.getLogger(__name__) # Use provided logger or create a default one
    current_logger.info(f"Searching for latest checkpoint in s3://{s3_bucket}/{s3_prefix}...")
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)

        latest_checkpoint_key = None
        latest_modified_time = None

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    object_key = obj["Key"]
                    # We are looking for checkpoint files created by save_checkpoint, not best models
                    # The pattern ensures we pick checkpoint_s3_epoch_N.pth or checkpoint_s3_epoch_N_batch_M.pth
                    # The `LastModified` time ensures we pick the truly latest one.
                    if object_key.startswith(s3_prefix) and object_key.endswith('.pth') and 'checkpoint_s3_epoch_' in object_key:
                        modified_time = obj["LastModified"]
                        
                        if latest_modified_time is None or modified_time > latest_modified_time:
                            latest_modified_time = modified_time
                            latest_checkpoint_key = object_key
        
        if latest_checkpoint_key:
            s3_path = f"s3://{s3_bucket}/{latest_checkpoint_key}"
            current_logger.info(f"Found latest checkpoint: {s3_path} (Last Modified: {latest_modified_time})")
            return s3_path
        else:
            current_logger.info(f"No checkpoint files found matching pattern '{checkpoint_filename_pattern}' in s3://{s3_bucket}/{s3_prefix}")
            return None

    except Exception as e:
        current_logger.error(f"Error listing S3 checkpoints from s3://{s3_bucket}/{s3_prefix}: {e}")
        return None


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        config: Configuration object
    
    Returns:
        torch.device
    """
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif config.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get the size of a model in megabytes.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        print(f"{self.name} took {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test AverageMeter
    print("\n1. Testing AverageMeter:")
    meter = AverageMeter('Loss')
    for i in range(10):
        meter.update(i * 0.1)
    print(meter)
    
    # Test accuracy
    print("\n2. Testing accuracy function:")
    output = torch.randn(32, 1000)
    target = torch.randint(0, 1000, (32,))
    top1, top5 = accuracy(output, target)
    print(f"Top-1 accuracy: {top1.item():.2f}%")
    print(f"Top-5 accuracy: {top5.item():.2f}%")
    
    # Test parameter counting
    print("\n3. Testing parameter counting:")
    from model import resnet50
    model = resnet50()
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    print(f"ResNet-50 parameters: {num_params:,}")
    print(f"ResNet-50 size: {model_size:.2f} MB")
    
    # Test timer
    print("\n4. Testing Timer:")
    with Timer("Test operation"):
        time.sleep(0.1)
    
    print("\nAll utility tests passed!")