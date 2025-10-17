# EC2_Upload_Test(MiniSet) Quick Start Guide

This guide details the steps to set up and run the ResNet-50 training script (`train_hf.py`) specifically within the `EC2_Upload_Test(MiniSet)` project on an AWS EC2 instance with GPU support.

---

## 🚀 Quick Testing on EC2

### Step 1: Launch EC2 Instance & SSH

*   **Recommended Instance Type**: `g4dn.2xlarge` (1 NVIDIA T4 GPU, 8 vCPU, 32 GB RAM) or similar.
*   **Recommended AMI**: Deep Learning AMI (Ubuntu) - PyTorch 2.0+ or a compatible Ubuntu AMI with NVIDIA drivers and CUDA installed.

    ```bash
    ssh -i your-key.pem ubuntu@your-ec2-ip
    ```
    *Replace `your-key.pem` with your private key path and `your-ec2-ip` with your instance's public IP.*

### Step 2: Set Up Project Environment

1.  **Clone the Repository (if not already done):**
    First, ensure you have the main `ResNet_50_Mini_Capstone` repository cloned. Navigate to your desired location (e.g., `/home/ubuntu/ERA`).
    https://github.com/erkarthi17/ERA/tree/2d614ab2316ff810c720769d4c527293cb1519fb/ResNet_50_Mini_Capstone/EC2_Upload_Test(MiniSet)
    ```bash
    cd ResNet_50_Mini_Capstone/EC2_Upload_Test\(MiniSet\)
    ```
    *If the repository is already cloned, just navigate to the `EC2_Upload_Test(MiniSet)` directory.*

2.  **Make Setup Script Executable and Run It:**
    The `setup_ec2.sh` script within *this directory* (`EC2_Upload_Test(MiniSet)`) will prepare your Python environment, including a virtual environment and necessary dependencies.
    ```bash
    chmod +x setup_ec2.sh
    bash ./setup_ec2.sh
    ```
    *Note: Using `bash ./setup_ec2.sh` is crucial to avoid `source: not found` errors.*

3.  **Activate Virtual Environment:**
    The script activates it, but for subsequent sessions, or if it deactivates, use:
    ```bash
    source venv/bin/activate
    ```

### Step 3: Data Preparation

This project (`EC2_Upload_Test(MiniSet)`) is designed to use a smaller dataset like `imagenette2`, which should be present within the `./imagenette2` sub-directory. The `data_hf.py` script automatically handles loading this data. You generally do **not** need to manually download ImageNet or specify `--data-root`.

### Step 4: Run a Quick Test (e.g., 2-5 epochs)

Execute the `train_hf.py` script. The script uses command-line arguments to override default settings defined in `config.py`.

```bash
python train_hf.py --epochs 2 --batch-size 64
```
*   Adjust `--epochs` for shorter or longer runs.
*   Adjust `--batch-size` based on your GPU memory.

### Step 5: Monitor Training (Optional, but Recommended)

Open a **new terminal** and SSH into your EC2 instance again to monitor progress.

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

Then, from the `EC2_Upload_Test(MiniSet)` directory (after activating venv if needed):

*   **Watch GPU Usage:**
    ```bash
    watch -n 1 nvidia-smi
    ```
*   **Monitor Logs:** (Logs are created in the parent `logs` directory as configured by `setup_ec2.sh`)
    ```bash
    tail -f ../logs/resnet50_imagenet_*.log
    ```
*   **Use TensorBoard:** (Access from your local machine's web browser)
    ```bash
    tensorboard --logdir=../logs --host=localhost --port=6006
    ```
    Access in your local browser: `http://your-ec2-ip:6006`

---

## 🎯 Different Test Scenarios

Once the quick test runs successfully, you can try longer runs.

*   **Scenario 1: Medium Test (5 epochs)**
    ```bash
    python train_hf.py --epochs 5 --batch-size 128
    ```

*   **Scenario 2: Realistic Test (10 epochs)**
    ```bash
    python train_hf.py --epochs 10 --batch-size 256
    ```

---

## ✅ After Successful Test

After your test completes successfully:

1.  **Check logs:** `cat ../logs/resnet50_imagenet_*.log`
2.  **Check checkpoints:** `ls -lh ../checkpoints/resnet50_imagenet/`
3.  **Verify accuracy:** Look for improvements in validation accuracy over epochs.

---
