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
    filename: str = 'checkpoint.pth'
):
    """
    Save training checkpoint.
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
        filename: Name of the checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save regular checkpoint
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    # Save best model
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filepath)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer=None, lr_scheduler=None):
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        lr_scheduler: Learning rate scheduler to load state into (optional)
    
    Returns:
        Dictionary containing loaded state
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best top-1 accuracy: {checkpoint.get('best_acc1', 0):.2f}%")
    
    return checkpoint


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