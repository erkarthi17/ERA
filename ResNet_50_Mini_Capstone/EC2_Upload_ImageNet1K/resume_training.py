import subprocess
import os

# --- CONFIG ---
EC2_USER = "ubuntu"
EC2_HOST = "18.133.77.135" # Ensure this is your current EC2 Public IP
KEY_PATH = "C:/Users/erkar/Downloads/ERAV4_EC2_V1_KEYPAIR.pem" # Ensure this path is correct on your local machine
PROJECT_PATH = "/home/ubuntu/ERA/ResNet_50_Mini_Capstone"
VENV_PATH = f"{PROJECT_PATH}/venv/bin/activate" # CORRECTED: Changed 'vevn' to 'venv'
CHECKPOINT_DIR = f"{PROJECT_PATH}/EC2_Upload_ImageNet1K/checkpoints"
TRAIN_SCRIPT = "EC2_Upload_ImageNet1K.train_s3"

# --- SSH Command Builder ---
def ssh_cmd(command):
    l_ssh_command = ["ssh", "-i", KEY_PATH,
        f"{EC2_USER}@{EC2_HOST}",
        command
    ]
    print("SSH Command is ", l_ssh_command)
    return l_ssh_command

# --- Find Latest Checkpoint ---
def get_latest_checkpoint():
    # Use -t for sorting by modification time (newest first) and head -n 1 to get the first one
    # Note: glob pattern must match the exact naming convention in your script
    cmd = f"ls -t {CHECKPOINT_DIR}/checkpoint_s3_epoch_*.pth | head -n 1"
    result = subprocess.run(ssh_cmd(cmd), capture_output=True, text=True)
    
    # Check for errors in the remote command execution
    if result.returncode != 0:
        print(f"❌ Error finding checkpoint on EC2: {result.stderr.strip()}")
        return None

    latest_ckpt = result.stdout.strip()
    return latest_ckpt if latest_ckpt else None # Return None if ls found nothing

# --- Resume Training ---
def resume_training():
    latest_ckpt = get_latest_checkpoint()
    if not latest_ckpt:
        print("❌ No checkpoint found. Starting new training run if no --resume is provided directly to train_s3.py.")
        # Optionally, you could add logic here to start a new training run
        # without --resume if no checkpoint is found.
        # For now, it will simply exit if no checkpoint is found.
        return

    train_cmd = (
        f"source {VENV_PATH} && " # Activate venv
        f"cd {PROJECT_PATH} && "    # Navigate to project root for python -m
        f"python -m {TRAIN_SCRIPT} "
        f"--resume {latest_ckpt} "
        f"--epochs 90 " # Ensure this matches your desired total epochs for the run
        f"--batch-size 256 " # Ensure this matches your optimal batch size
        f"--lr 0.003 "       # Ensure this matches your optimal LR
        f"--s3-bucket imagenet-dataset-karthick-kannan "
        f"--s3-prefix-train imagenet-1k/train/ "
        f"--s3-prefix-val imagenet-1k/validation/"
    )

    print(f"🚀 Resuming training from: {latest_ckpt}")
    print(f"Executing remote command: {train_cmd}") # Added for better debugging
    subprocess.run(ssh_cmd(train_cmd))

# --- Trigger ---
if __name__ == "__main__":
    resume_training()