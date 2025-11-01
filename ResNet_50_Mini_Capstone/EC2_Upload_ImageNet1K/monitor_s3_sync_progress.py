#!/usr/bin/env python3
"""
Monitor S3 → local sync progress and estimate remaining data + ETA.
Works even while `aws s3 sync` is running in another terminal.
"""

import boto3
import os
import time
from tqdm import tqdm
from botocore.exceptions import NoCredentialsError, ClientError

# ====== CONFIGURE THIS SECTION ======
S3_BUCKET = "imagenet-dataset-karthick-kannan"
S3_PREFIX = "imagenet-1k/validation/"   # Change to 'validation/' to check val data
LOCAL_DIR = "/home/ubuntu/imagenet_local/validation"
REGION = "us-east-1"  # Optional, only if your EC2 isn't in same region
CHECK_INTERVAL = 60   # seconds between checks
# ====================================


def get_s3_total_size(bucket, prefix):
    """Return total size (in bytes) and number of files in an S3 prefix."""
    s3 = boto3.client("s3", region_name=REGION)
    paginator = s3.get_paginator("list_objects_v2")

    total_size = 0
    total_files = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            total_size += obj["Size"]
            total_files += 1

    return total_size, total_files


def get_local_total_size(local_dir):
    """Return total size (in bytes) and number of files in a local directory."""
    total_size = 0
    total_files = 0
    for root, _, files in os.walk(local_dir):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_size += os.path.getsize(fp)
                total_files += 1
            except FileNotFoundError:
                continue
    return total_size, total_files


def human_readable_size(size_bytes):
    """Convert bytes → human-readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024


def monitor_sync():
    print(f"🔍 Monitoring sync progress from s3://{S3_BUCKET}/{S3_PREFIX} → {LOCAL_DIR}")
    print("Press Ctrl+C to stop.\n")

    try:
        s3_total_size, s3_total_files = get_s3_total_size(S3_BUCKET, S3_PREFIX)
    except (NoCredentialsError, ClientError) as e:
        print(f"❌ AWS credentials or permissions error: {e}")
        return

    print(f"📦 S3 Total: {human_readable_size(s3_total_size)} across {s3_total_files:,} files.\n")

    start_time = time.time()
    prev_local_size = 0

    try:
        while True:
            local_size, local_files = get_local_total_size(LOCAL_DIR)
            percent = (local_size / s3_total_size) * 100
            remaining = s3_total_size - local_size

            elapsed = time.time() - start_time
            delta = local_size - prev_local_size
            speed = delta / elapsed if elapsed > 0 else 0  # bytes/sec
            eta_seconds = remaining / speed if speed > 0 else float("inf")

            # Format ETA
            eta_h = int(eta_seconds // 3600)
            eta_m = int((eta_seconds % 3600) // 60)
            eta_s = int(eta_seconds % 60)

            tqdm.write(
                f"[{time.strftime('%H:%M:%S')}] "
                f"{percent:.2f}% synced | "
                f"Local: {human_readable_size(local_size)} / {human_readable_size(s3_total_size)} "
                f"({local_files:,}/{s3_total_files:,} files) | "
                f"Speed: {human_readable_size(speed)}/s | "
                f"ETA: {eta_h}h {eta_m}m {eta_s}s"
            )

            prev_local_size = local_size
            start_time = time.time()
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 Stopped monitoring.")
        local_size, local_files = get_local_total_size(LOCAL_DIR)
        percent = (local_size / s3_total_size) * 100
        print(f"✅ Current Sync Progress: {percent:.2f}% ({local_files:,}/{s3_total_files:,} files)")



if __name__ == "__main__":
    monitor_sync()