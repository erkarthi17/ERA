import subprocess
import os

# --- CONFIG ---
EC2_USER = "ubuntu"
EC2_HOST = "3.9.139.182"
KEY_PATH = "C:/Users/erkar/Downloads/ERAV4_EC2_V1_KEYPAIR.pem"
PROJECT_PATH = "/home/ubuntu/ERA/ResNet_50_Mini_Capstone"
VENV_PATH = f"{PROJECT_PATH}/vevn/bin/activate"
CHECKPOINT_DIR = f"{PROJECT_PATH}/EC2_Upload_ImageNet1K/checkpoints"
TRAIN_SCRIPT = "EC2_Upload_ImageNet1K.train_s3"

# --- SSH Command Builder ---
def ssh_cmd(command):
    return [
        "ssh", "-i", KEY_PATH,
        f"{EC2_USER}@{EC2_HOST}",
        command
    ]

# --- Find Latest Checkpoint ---
def get_latest_checkpoint():
    cmd = f"ls -t {CHECKPOINT_DIR}/checkpoint_s3_epoch_*.pth | head -n 1"
    result = subprocess.run(ssh_cmd(cmd), capture_output=True, text=True)
    latest_ckpt = result.stdout.strip()
    return latest_ckpt

# --- Resume Training ---
def resume_training():
    latest_ckpt = get_latest_checkpoint()
    if not latest_ckpt:
        print("❌ No checkpoint found.")
        return

    train_cmd = (
        f"source {VENV_PATH} && "
        f"cd {PROJECT_PATH} && "
        f"python -m {TRAIN_SCRIPT} "
        f"--resume {latest_ckpt} "
        f"--epochs 90 "
        f"--batch-size 256 "
        f"--lr 0.003 "
        f"--s3-bucket imagenet-dataset-karthick-kannan "
        f"--s3-prefix-train imagenet-1k/train/ "
        f"--s3-prefix-val imagenet-1k/validation/"
    )

    print(f"🚀 Resuming training from: {latest_ckpt}")
    subprocess.run(ssh_cmd(train_cmd))

# --- Trigger ---
if __name__ == "__main__":
    resume_training()