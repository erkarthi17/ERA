import os
import io
import boto3
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from threading import Lock

# 🔧 Config
HF_DATASET = "imagenet-1k"
S3_BUCKET = "imagenet-dataset-karthick-kannan"
S3_PREFIX = "imagenet-1k/test"
MAX_WORKERS = 12  # Good for t3.xlarge
CHECKPOINT_FILE = "checkpoint_test.txt"

# 🧠 Auth
HF_TOKEN = os.getenv("HF_TOKEN")  # Or paste your token directly
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
def upload_if_missing(i, ex, uploaded_keys):
    global upload_count
    key = f"{S3_PREFIX}/{i:07d}.jpg"
    if key in uploaded_keys:
        return

    img = ex["image"]
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG", quality=90)
    img_bytes.seek(0)

    try:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=img_bytes.getvalue())
        save_to_checkpoint(key)
        with lock:
            upload_count += 1
            if upload_count % 1000 == 0:
                print(f"✅ Uploaded {upload_count} test images so far...")
    except Exception as e:
        print(f"❌ Failed to upload {key}: {e}")

# 🧵 Main loop
uploaded_keys = load_checkpoint()
print(f"\n🔄 Starting ImageNet test upload...")

ds = load_dataset(HF_DATASET, split="test", streaming=True, token=HF_TOKEN)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []
    for i, ex in enumerate(ds):
        futures.append(executor.submit(upload_if_missing, i, ex, uploaded_keys))
        if i % 1000 == 0:
            print(f"📤 Queued {i} test uploads...")

    for f in tqdm(futures, desc="Uploading test split"):
        f.result()

print("\n✅ Test upload complete. Checkpoint updated.")
