<<<<<<< HEAD
import os
import io
import boto3
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from threading import Lock

# 🔧 Config
HF_DATASET = "imagenet-1k"
SPLITS = ["train", "validation"]
S3_BUCKET = "imagenet-dataset-karthick-kannan"
S3_PREFIX = "imagenet-1k"
MAX_WORKERS = 12  # Recommended for t3.xlarge
CHECKPOINT_FILE = "checkpoint.txt"
=======
from datasets import load_dataset
import boto3, io, os
from PIL import Image
from tqdm import tqdm

HF_DATASET = "ILSVRC/imagenet-1k"
S3_BUCKET  = "imagenet-dataset-karthick-kannan"
S3_PREFIX  = "imagenet1k"
SPLITS     = ["train", "validation"]
>>>>>>> parent of 2b73e8d (Update download_imagenet_to_s3.py to use environment variable for Hugging Face token)

# 🧠 Auth
HF_TOKEN = os.getenv("HF_TOKEN")
s3 = boto3.client("s3")

# 📦 Load checkpoint
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return set(line.strip() for line in f)

# 📝 Save to checkpoint
def save_to_checkpoint(key):
    with lock:
        with open(CHECKPOINT_FILE, "a") as f:
            f.write(f"{key}\n")

# 🔄 Upload counter
upload_count = 0
lock = Lock()

# 🚀 Upload function
def upload_if_missing(i, ex, split, uploaded_keys):
    global upload_count
    label = ex["label"]
    key = f"{S3_PREFIX}/{split}/{label}/{i:07d}.jpg"
    if key in uploaded_keys:
        return

    img = ex["image"]
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")  # 💡 Strip alpha channel

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG", quality=90)
    img_bytes.seek(0)

    try:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=img_bytes.getvalue())
        save_to_checkpoint(key)
        with lock:
            upload_count += 1
            if upload_count % 1000 == 0:
                print(f"✅ Uploaded {upload_count} files so far...")
    except Exception as e:
        print(f"❌ Failed to upload {key}: {e}")


# 🧵 Main loop
uploaded_keys = load_checkpoint()

for split in SPLITS:
<<<<<<< HEAD
    print(f"\n🔄 Processing split: {split}")
    ds = load_dataset(HF_DATASET, split=split, streaming=True, token=HF_TOKEN)
=======
    ds = load_dataset(HF_DATASET, split=split, streaming=True, use_auth_token=True)
    print(f"Uploading {split} split...")
>>>>>>> parent of 2b73e8d (Update download_imagenet_to_s3.py to use environment variable for Hugging Face token)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, ex in enumerate(ds):
            futures.append(executor.submit(upload_if_missing, i, ex, split, uploaded_keys))
            if i % 1000 == 0:
                print(f"📤 Queued {i} uploads...")

        for f in tqdm(futures, desc=f"Uploading {split}"):
            f.result()

print("\n✅ Parallel upload complete. Checkpoint updated.")
