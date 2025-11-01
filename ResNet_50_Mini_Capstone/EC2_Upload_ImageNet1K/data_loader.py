# data_loader.py
# Unified data loader entry point — chooses S3 or EBS backend

from .data_s3 import get_data_loaders as get_s3_data_loaders
from .data_ebs import get_data_loaders as get_ebs_data_loaders


def get_data_loaders(config):
    """
    Unified entrypoint for data loading.

    Uses S3 or EBS backend based on config.data_source.
    """
    source = getattr(config, "data_source", "s3").lower()

    if source == "s3":
        return get_s3_data_loaders(config)
    elif source == "ebs":
        return get_ebs_data_loaders(config)
    else:
        raise ValueError(f"Unsupported data source: {source}")