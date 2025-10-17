# 🧠 ResNet-50 Training from Scratch on AWS EC2 (MiniSet Credit Submission)

This project implements a **ResNet-50** model trained **from scratch** on a small subset of ImageNet (**Imagenette2**) using **PyTorch**.  
It serves as the **credit request submission** for the “ResNet-50 from Scratch on ImageNet 1K” assignment under TSAI, demonstrating a complete, functional deep learning pipeline on an AWS GPU instance.

---

## 🚀 Project Overview

**Core Highlights:**

- 🧩 **ResNet-50 Architecture** — Implemented from scratch in PyTorch.  
- 🔄 **Training from Scratch** — No pre-trained weights are used.  
- ⚙️ **PyTorch Framework** — End-to-end training and evaluation pipeline.  
- 📦 **Hugging Face Datasets Integration** — Efficient dataset handling via `data_hf.py`.  
- ☁️ **AWS EC2 Optimized** — Automated setup via `setup_ec2.sh` for GPU-backed instances (e.g., `g4dn.2xlarge`).  
- ⚡ **Mixed Precision Training** — Supports NVIDIA Apex or PyTorch AMP for faster and memory-efficient training.  
- 🧾 **Logging & Checkpointing** — Saves model states and logs for resumption and monitoring.

---

## 🎯 Assignment Context

This submission demonstrates the ability to:
1. Set up and train a **ResNet-50 model from scratch** on EC2.
2. Achieve a working end-to-end training pipeline using Imagenette2 as a mini ImageNet subset.

Upon receiving AWS credits, the next goal is to **scale training to the full ImageNet-1K dataset**, targeting:
- 🎯 **75% Top-1 accuracy**
- ⭐ **81% Top-1 accuracy** (for bonus points)

> **Note:** No pre-trained weights or transfer learning techniques are used — this is a pure “train from scratch” setup.

---

## 🧰 Setup & Usage Guide

### ✅ Prerequisites

- AWS **EC2 instance** with GPU (Recommended: `g4dn.2xlarge`)
- **Deep Learning AMI (Ubuntu)** with CUDA and PyTorch pre-installed  
- **Git** and **SSH access** configured
- **Python 3.8+**

---

### 1️⃣ Clone the Repository

```bash
# If not already cloned
git clone https://github.com/erkarthi17/ERA/tree/6a73600b8317b0b3aa690a2c0d899549452c22a3/ResNet_50_Mini_Capstone/EC2_Upload_Test(MiniSet)

# Navigate to this project directory
cd ResNet_50_Mini_Capstone/EC2_Upload_Test\(MiniSet\)
```

---

### 2️⃣ Set Up Your Environment

Run the setup script to configure everything automatically:

```bash
chmod +x setup_ec2.sh
bash ./setup_ec2.sh
```

This script will:
- Update system packages
- Install Python, `pip`, and `venv`
- Create and activate a virtual environment (`venv`)
- Install dependencies from `requirements.txt`
- Set up `checkpoints/`, `logs/`, and `data/` directories

> ⚠️ Always use `bash ./setup_ec2.sh` (not `sh`) to ensure correct execution.

---

### 3️⃣ Activate Virtual Environment (for New Sessions)

```bash
source venv/bin/activate
```

---

### 4️⃣ Data Setup

This project uses **Imagenette2**, a lightweight subset of ImageNet.  
Place the dataset inside the `./imagenette2` folder — the `data_hf.py` loader will automatically detect it.

> 💡 No need to manually download ImageNet or specify `--data-root`.

---

### 5️⃣ Run a Quick Training Test

Run a short training cycle (2–5 epochs) to verify your setup:

```bash
python train_hf.py --epochs 2 --batch-size 64
```

- Adjust `--epochs` for longer runs (e.g., `--epochs 5`)
- Adjust `--batch-size` based on GPU memory

---

### 6️⃣ Monitor Training Progress

In a **new SSH terminal**, connect to your EC2 instance:

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

Then monitor training via:

#### 🧠 GPU Usage
```bash
watch -n 1 nvidia-smi
```

#### 📝 Live Logs
```bash
tail -f ../logs/resnet50_imagenet_*.log
```

#### 📊 TensorBoard (optional)
```bash
tensorboard --logdir=../logs --host=localhost --port=6006
```
Access from your browser at:  
👉 `http://<your-ec2-ip>:6006`

---

## 📂 Repository Structure

```bash
.
├── config.py              # Configuration settings
├── data_hf.py             # Hugging Face dataset loader
├── imagenette2/           # Imagenette2 dataset folder
├── model.py               # ResNet-50 model definition
├── requirements.txt       # Dependencies list
├── setup_ec2.sh           # EC2 environment setup script
├── train_hf.py            # Main training script
├── utils.py               # Logging, metrics, checkpointing utilities
└── README.md              # Project documentation (this file)
```

---

## 🧾 Submission Components

| Component | Description |
|------------|-------------|
| 🧠 **GitHub Repo Link** | https://github.com/erkarthi17/ERA/tree/6a73600b8317b0b3aa690a2c0d899549452c22a3/ResNet_50_Mini_Capstone/EC2_Upload_Test(MiniSet) |
| 🤗 **Hugging Face Spaces Link** | Will be added post ImageNet 1K Build |
| 🎥 **YouTube Demo Video** | Same as above |
| 📜 **Markdown Log File** | `FULL_TRAINING_LOGS.md` — consolidated logs from all epochs |
| 🖥️ **EC2 Screenshot** | Attached in the Mail |

---

## 🧩 Quick Command Reference

```bash
# Setup
chmod +x setup_ec2.sh
bash ./setup_ec2.sh

# Activate Environment
source venv/bin/activate

# Run Training
python train_hf.py --epochs 2 --batch-size 64

# Monitor GPU
watch -n 1 nvidia-smi

# View Logs
tail -f ../logs/resnet50_imagenet_*.log
```

---

## 🏁 Next Steps

Once AWS credits are approved:
- Scale up to **ImageNet-1k**
- Increase training duration & hyperparameter tuning
- Log results for full-credit submission

---

**Author(s):** Karthick Kannan
**Institution:** TSAI – The School of AI  
**Project:** ResNet-50 from Scratch – EC2 MiniSet Credit Submission