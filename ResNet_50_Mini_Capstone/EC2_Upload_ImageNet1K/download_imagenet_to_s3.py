from datasets import load_dataset
import boto3, io, os
from PIL import Image
from tqdm import tqdm
import os

HF_TOKEN = os.getenv("HF_TOKEN", "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") 


HF_DATASET = "ILSVRC/imagenet-1k"
S3_BUCKET  = "imagenet-dataset-karthick-kannan"
S3_PREFIX  = "imagenet1k"
SPLITS     = ["train", "validation"]

s3 = boto3.client("s3")

for split in SPLITS:
    ds = load_dataset(HF_DATASET, split=split, streaming=True, token=HF_TOKEN)
    print(f"Uploading {split} split...")

    for i, ex in enumerate(tqdm(ds, total=None)):
        label = ex["label"]
        img   = ex["image"]
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=90)
        img_bytes.seek(0)

        key = f"{S3_PREFIX}/{split}/{label}/{i:07d}.jpg"
        s3.upload_fileobj(img_bytes, S3_BUCKET, key)

print("✅ Upload complete.")