import boto3
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import NoCredentialsError, ClientError

# ====== CONFIGURE THIS SECTION ======
S3_BUCKET = "imagenet-dataset-karthick-kannan"
S3_PREFIXES = {
    "train": "imagenet-1k/train/",
    "val": "imagenet-1k/validation/"
}
LOCAL_DIRS = {
    "train": "/home/ubuntu/imagenet_local/train",
    "val": "/home/ubuntu/imagenet_local/validation"
}
REGION = "eu-west-2"
MAX_WORKERS = 16
CHECK_INTERVAL = 60  # seconds
# ====================================

s3 = boto3.client("s3", region_name=REGION)

def list_s3_files(bucket, prefix):
    """List all files under a prefix in S3."""
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            files.append(obj["Key"])
    return files

def download_file(key, prefix, local_root):
    """Download a single file from S3 to local path."""
    local_path = os.path.join(local_root, os.path.relpath(key, prefix))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        s3.download_file(S3_BUCKET, key, local_path)
    except Exception as e:
        tqdm.write(f"❌ Failed to download {key}: {e}")

def get_local_total_size(local_dir):
    """Return total size and file count of a local directory."""
    total_size = 0
    total_files = 0
    for root, _, files in os.walk(local_dir):
        for f in files:
            try:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
                total_files += 1
            except FileNotFoundError:
                continue
    return total_size, total_files

def human_readable_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

def monitor_and_sync():
    print("🔁 Starting sync for both training and validation sets...\n")

    all_tasks = []
    total_files = 0
    for split in ["train", "val"]:
        prefix = S3_PREFIXES[split]
        local_dir = LOCAL_DIRS[split]
        keys = list_s3_files(S3_BUCKET, prefix)
        total_files += len(keys)
        for key in keys:
            all_tasks.append((key, prefix, local_dir))

    print(f"📦 Total files to sync: {total_files:,}\n")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_file, key, prefix, local_dir) for key, prefix, local_dir in all_tasks]

        with tqdm(total=len(futures), desc="Sync Progress", unit="file", dynamic_ncols=True) as pbar:
            completed = set()
            while True:
                for i, f in enumerate(futures):
                    if f.done() and i not in completed:
                        completed.add(i)
                        pbar.update(1)

                # Optional: show speed and ETA
                train_size, _ = get_local_total_size(LOCAL_DIRS["train"])
                val_size, _ = get_local_total_size(LOCAL_DIRS["val"])
                local_size = train_size + val_size
                elapsed = time.time() - start_time
                speed = local_size / elapsed if elapsed > 0 else 0
                remaining = total_files - len(completed)
                eta_seconds = (remaining * (elapsed / len(completed))) if completed else float("inf")
                eta_m = int((eta_seconds % 3600) // 60)
                eta_s = int(eta_seconds % 60)

                tqdm.write(
                    f"[{time.strftime('%H:%M:%S')}] {len(completed):,}/{total_files:,} files | "
                    f"Speed: {human_readable_size(speed)}/s | ETA: {eta_m}m {eta_s}s"
                )

                if len(completed) == len(futures):
                    break
                time.sleep(CHECK_INTERVAL)

    print("✅ Sync complete for both training and validation sets.")

if __name__ == "__main__":
    monitor_and_sync()