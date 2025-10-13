"""
Configuration file for ResNet-50 ImageNet training.
All hyperparameters and paths are defined here.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Training configuration for ResNet-50 on ImageNet."""
    
    # Model parameters
    model_name: str = "resnet50"
    num_classes: int = 1000
    
    # Data parameters
    data_root: str = "/path/to/imagenet"  # Update this path
    train_dir: str = "train"
    val_dir: str = "val"
    num_workers: int = 8
    pin_memory: bool = True
    
    # Training parameters
    epochs: int = 90
    batch_size: int = 256  # Effective batch size
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Learning rate schedule
    lr_scheduler: str = "step"  # 'step' or 'cosine'
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    warmup_epochs: int = 5
    warmup_lr: float = 0.01
    
    # Data augmentation
    image_size: int = 224
    min_crop_size: float = 0.08
    color_jitter: float = 0.4
    interpolation: str = "bilinear"
    
    # Validation
    val_batch_size: int = 256
    val_resize_size: int = 256
    val_center_crop_size: int = 224
    
    # Optimization
    optimizer: str = "sgd"  # 'sgd' or 'adamw'
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    
    # Logging and checkpointing
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    checkpoint_name: str = "resnet50_imagenet"
    save_every: int = 10  # Save checkpoint every N epochs
    log_interval: int = 100  # Log every N batches
    eval_interval: int = 1  # Evaluate every N epochs
    
    # Resume training
    resume: bool = False
    resume_path: Optional[str] = None
    
    # Hardware
    device: str = "cuda"  # 'cuda', 'cpu', or 'mps'
    mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0 compile
    
    # Distributed training (for multi-GPU)
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    
    # Seed for reproducibility
    seed: int = 42
    
    # Print config
    print_freq: int = 100
    
    def __post_init__(self):
        """Validate and set derived parameters."""
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set data paths
        self.train_path = os.path.join(self.data_root, self.train_dir)
        self.val_path = os.path.join(self.data_root, self.val_dir)
        
        # Validate paths
        if not os.path.exists(self.train_path):
            print(f"Warning: Training data path does not exist: {self.train_path}")
        if not os.path.exists(self.val_path):
            print(f"Warning: Validation data path does not exist: {self.val_path}")
    
    def __str__(self):
        """String representation of config."""
        attrs = []
        for key, value in self.__dict__.items():
            attrs.append(f"  {key}: {value}")
        return "Config:\n" + "\n".join(attrs)


# Default configuration
default_config = Config()


if __name__ == "__main__":
    # Print default configuration
    config = Config()
    print(config)
    
    # Example: Custom configuration
    print("\n" + "="*50)
    print("Custom Configuration Example:")
    print("="*50)
    custom_config = Config(
        batch_size=128,
        learning_rate=0.05,
        epochs=100,
        data_root="/data/imagenet"
    )
    print(custom_config)

