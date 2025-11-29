"""
Upload Stock Market BPE Tokenizer to HuggingFace
"""

import os
from huggingface_hub import HfApi, create_repo, login

# Configuration
print("="*50)
print("HuggingFace Uploader")
print("="*50)

# 1. Login
print("\nPlease enter your HuggingFace Write Token.")
print("(Get it from: https://huggingface.co/settings/tokens)")
token = input("Token: ").strip()

try:
    login(token=token)
    print("✓ Login successful!")
except Exception as e:
    print(f"❌ Login failed: {e}")
    exit(1)

# 2. Get Username
api = HfApi()
user_info = api.whoami()
username = user_info['name']
print(f"Logged in as: {username}")

# 3. Repository Config
MODEL_NAME = "stock-market-bpe-tokenizer"
REPO_ID = f"{username}/{MODEL_NAME}"

print(f"\nPreparing to upload to {REPO_ID}...")

try:
    # 4. Create Repository (if it doesn't exist)
    print("Creating repository...")
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    
    # 5. Upload Files
    files_to_upload = [
        "stock_bpe.merges",
        "stock_bpe.vocab",
        "tokenizer.py",
        "README.md",
        "example_usage.ipynb",
        "requirements.txt"
    ]
    
    print("\nUploading files...")
    for file in files_to_upload:
        if os.path.exists(file):
            print(f"Uploading {file}...")
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=REPO_ID,
                repo_type="model"
            )
            print(f"✓ {file} uploaded")
        else:
            print(f"⚠️ Warning: {file} not found, skipping")
            
    print("\n" + "="*50)
    print("🎉 Upload Complete!")
    print("="*50)
    print(f"Your model is live at: https://huggingface.co/{REPO_ID}")
    print("\nDon't forget to add this link to your assignment submission!")

except Exception as e:
    print(f"\n❌ Error: {e}")
