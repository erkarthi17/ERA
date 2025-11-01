import os
from dataclasses import dataclass, field
from typing import Optional

from .base_config import Config as BaseConfig

@dataclass
class Config(BaseConfig):
    """
    Configuration for ResNet-50 ImageNet training.
    Supports both S3 and EBS data sources, switchable via data_source flag.
    """
    # 🔀 Data source switch: 's3' or 'ebs'
    data_source: str = field(default="s3", metadata={"help": "Data source for ImageNet (options: 's3', 'ebs')"})

    # ---- S3 configuration ----
    data_root: str = "s3"
    s3_bucket: str = "imagenet-dataset-karthick-kannan"
    s3_prefix_train: str = "imagenet-1k/train/"
    s3_prefix_val: str = "imagenet-1k/validation/"
    s3_region: Optional[str] = None

    # ---- EBS configuration ----
    # Root mount path for EBS (adjust this based on your instance)
    ebs_root: str = "/home/ubuntu/imagenet_local"
    ebs_train_dir: str = "train"
    ebs_val_dir: str = "val"

    # ---- General training subsets ----
    train_subset_size: Optional[int] = None
    val_subset_size: Optional[int] = None

    # ---- Checkpointing ----
    save_every: int = 1
    eval_interval: int = 1
    save_every_n_batches: Optional[int] = 500

    s3_checkpoint_bucket: str = "imagenet-dataset-karthick-kannan"
    s3_checkpoint_prefix: str = "resnet50_checkpoints/"
    resume_from_s3_latest: bool = False

    # ---- Caching & timeout options ----
    cache_dir: str = field(default=".s3_cache")
    force_relist_s3: bool = False
    dataloader_timeout: int = 900
    dataloader_retry_limit: int = 3
    s3_max_retries: int = 5
    s3_request_timeout: int = 60
    s3_socket_timeout: int = 60

    def __post_init__(self):
        base_dir = os.path.dirname(__file__)
        self.checkpoint_dir = os.path.join(base_dir, "checkpoints")
        self.log_dir = os.path.join(base_dir, "logs")
        self.cache_dir = os.path.join(base_dir, self.cache_dir)

        for d in [self.checkpoint_dir, self.log_dir, self.cache_dir]:
            os.makedirs(d, exist_ok=True)

        self._build_paths()

    def _build_paths(self):
        # Construct paths dynamically based on source
        if self.data_source.lower() == "s3":
            self.train_path = f"s3://{self.s3_bucket}/{self.s3_prefix_train}"
            self.val_path = f"s3://{self.s3_bucket}/{self.s3_prefix_val}"
        elif self.data_source.lower() == "ebs":
            self.train_path = os.path.join(self.ebs_root, self.ebs_train_dir)
            self.val_path = os.path.join(self.ebs_root, self.ebs_val_dir)
        else:
            raise ValueError(f"Unsupported data_source: {self.data_source}")

        self.s3_checkpoint_path = f"s3://{self.s3_checkpoint_bucket}/{self.s3_checkpoint_prefix}" 

    def __str__(self):
        return (
            f"Config (Data Source: {self.data_source})\n" +
            "\n".join([f"{k}: {v}" for k, v in self.__dict__.items()])
        )

default_config = Config()

if __name__ == "__main__":
    print(Config())