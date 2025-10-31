import os
from dataclasses import dataclass, field
from typing import Optional

from .base_config import Config as BaseConfig


@dataclass
class Config(BaseConfig):
    """
    Configuration for ResNet-50 ImageNet training with S3 data loading.
    Inherits from the base Config and overrides data-related parameters.
    """
    # Override data_root to indicate S3 usage (though not directly used for path here)
    data_root: str = "s3"

    # S3 specific parameters
    s3_bucket: str = "imagenet-dataset-karthick-kannan"
    s3_prefix_train: str = "imagenet-1k/train/"  # Path within S3 bucket for training data
    s3_prefix_val: str = "imagenet-1k/validation/"  # Path within S3 bucket for validation data

    # You might need to specify region if not configured globally on EC2
    s3_region: Optional[str] = None  # e.g., "us-east-1"

    # Hugging Face dataset name (still used if you want to use HF's image processing, etc.)
    # But primary image loading will be from S3
    hf_dataset_name: str = "imagenet-1k"

    # --- Added/Modified for Checkpointing ---
    # Save checkpoint every N epochs
    save_every: int = 1
    # Evaluate model every N epochs (important for 'is_best' checkpointing)
    eval_interval: int = 1
    # New: Save checkpoint every N batches within an epoch
    save_every_n_batches: Optional[int] = 500 # Set to an integer like 500 to enable batch-level saving
    # --- End of Checkpointing changes ---

       # --- Dataset subset configuration (optional, used for debugging/sanity runs) ---
    train_subset_size: Optional[int] = None  # e.g., 10000 for small-train debugging
    val_subset_size: Optional[int] = None    # e.g., 1000 for small-val debugging

    # --- S3 Caching Configuration (New) ---
    cache_dir: str = field(default=".s3_cache", metadata={"help": "Directory for S3 file list cache"})
    force_relist_s3: bool = field(default=False, metadata={"help": "Force re-listing S3 files, ignoring cache"})
    # --- End of S3 Caching Configuration ---

    # --- S3 Checkpoint Configuration (New) ---
    s3_checkpoint_bucket: str = "imagenet-dataset-karthick-kannan" # Or a dedicated bucket for checkpoints
    s3_checkpoint_prefix: str = "resnet50_checkpoints/" # Prefix within the bucket where checkpoints will be stored
    resume_from_s3_latest: bool = False # Flag to automatically find and resume from latest S3 checkpoint
    # --- End of S3 Checkpoint Configuration ---


    def __post_init__(self):
        """Validate and set derived parameters."""
        # Ensure checkpoint and log directories exist relative to the current script's directory
        # This will place 'checkpoints' and 'logs' folders directly inside 'EC2_Upload_ImageNet1K'
        base_dir = os.path.dirname(__file__)
        self.checkpoint_dir = os.path.join(base_dir, "checkpoints")
        self.log_dir = os.path.join(base_dir, "logs")
        self.cache_dir = os.path.join(base_dir, self.cache_dir) # Make cache_dir relative to base_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True) # Ensure cache directory exists
        
        # S3 paths will be constructed in data_s3.py
        self.train_path = f"s3://{self.s3_bucket}/{self.s3_prefix_train}"
        self.val_path = f"s3://{self.s3_bucket}/{self.s3_prefix_val}"
        
        # New: Create S3 paths for checkpoints
        self.s3_checkpoint_path = f"s3://{self.s3_checkpoint_bucket}/{self.s3_checkpoint_prefix}"

                # Ensure optional attributes exist to avoid AttributeError in training script
        if not hasattr(self, "train_subset_size"):
            self.train_subset_size = None
        if not hasattr(self, "val_subset_size"):
            self.val_subset_size = None


        # The base config's path validation will warn, but we'll handle S3 paths in data_s3.py
        # You can remove these warnings if they are not relevant for S3 paths
        # if not os.path.exists(self.train_path):
        #     print(f"Warning: Training data path does not exist: {self.train_path}")
        # if not os.path.exists(self.val_path):
        #     print(f"Warning: Validation data path does not exist: {self.val_path}")

    def __str__(self):
        """String representation of config."""
        attrs = []
        for key, value in self.__dict__.items():
            # Exclude BaseConfig's redundant data_root if it's the default 'data'
            # Or customize this further to show only relevant S3-specific fields
            attrs.append(f"  {key}: {value}")
        
        # Also include fields from BaseConfig if needed, but ensure no duplication
        # This approach ensures all fields in *this* Config are included.
        return "Config (S3):\\\n" + "\\\n".join(attrs)


# Default configuration for S3
default_config = Config()

if __name__ == "__main__":
    config = Config()
    print(config)