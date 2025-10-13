# ResNet-50 ImageNet Training from Scratch

This repository contains a complete implementation for training ResNet-50 from scratch on the ImageNet dataset. The code is designed to run on AWS EC2, SageMaker, or local machines.

## 📋 Project Structure

```
ResNet_50_Mini_Capstone/
├── model.py              # ResNet-50 architecture implementation
├── config.py             # Configuration and hyperparameters
├── data.py               # Data loading and preprocessing
├── train.py              # Main training script
├── utils.py              # Utility functions (logging, metrics, etc.)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── checkpoints/         # Saved model checkpoints (created during training)
    └── resnet50_imagenet/
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare ImageNet Dataset

Download and extract the ImageNet dataset. The expected directory structure is:

```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### 3. Configure Training

Edit `config.py` to set your data path and hyperparameters:

```python
config = Config(
    data_root="/path/to/imagenet",
    batch_size=256,
    learning_rate=0.1,
    epochs=90,
    # ... other parameters
)
```

### 4. Run Training

```bash
# Basic training
python train.py --data-root /path/to/imagenet

# Custom batch size and learning rate
python train.py --data-root /path/to/imagenet --batch-size 128 --lr 0.05

# Resume from checkpoint
python train.py --data-root /path/to/imagenet --resume checkpoints/resnet50_imagenet/checkpoint_epoch_30.pth
```

## 🔧 Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 256 | Training batch size |
| `learning_rate` | 0.1 | Initial learning rate |
| `epochs` | 90 | Number of training epochs |
| `lr_scheduler` | "step" | Learning rate scheduler |
| `lr_step_size` | 30 | Epochs between LR decay |
| `lr_gamma` | 0.1 | LR decay factor |
| `weight_decay` | 1e-4 | Weight decay (L2 regularization) |
| `momentum` | 0.9 | SGD momentum |
| `mixed_precision` | True | Use mixed precision training |
| `num_workers` | 8 | Data loading workers |

## 📊 Training Features

- ✅ **ResNet-50 Architecture**: Implemented from scratch (no pretrained weights)
- ✅ **Mixed Precision Training**: Automatic mixed precision for faster training
- ✅ **Learning Rate Scheduling**: Step decay or cosine annealing
- ✅ **Data Augmentation**: Random resized crop, horizontal flip, color jitter
- ✅ **Checkpointing**: Save and resume training
- ✅ **Logging**: TensorBoard and file logging
- ✅ **Validation**: Top-1 and Top-5 accuracy tracking
- ✅ **Progress Tracking**: Real-time training metrics

## 🖥️ Running on AWS EC2

### 1. Launch EC2 Instance

```bash
# Recommended: p3.2xlarge or p3.8xlarge (GPU instances)
# AMI: Deep Learning AMI (Ubuntu) or Deep Learning Base AMI
```

### 2. Setup Environment

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Clone repository
git clone <your-repo-url>
cd ResNet_50_Mini_Capstone

# Install dependencies
pip install -r requirements.txt
```

### 3. Download ImageNet

```bash
# Option 1: Download from official source
# Visit: https://www.image-net.org/download.php

# Option 2: Use pre-downloaded data from S3
aws s3 sync s3://your-bucket/imagenet /data/imagenet
```

### 4. Start Training

```bash
# Run training (use nohup for background execution)
nohup python train.py \
    --data-root /data/imagenet \
    --batch-size 256 \
    --epochs 90 \
    > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### 5. Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006

# Check GPU usage
watch -n 1 nvidia-smi
```

## 🧪 Testing on Small Dataset

Before full training, test on a subset:

```bash
# Create a small subset for testing
mkdir -p /tmp/imagenet_subset/{train,val}
# Copy a few classes from ImageNet

# Run a short training test
python train.py \
    --data-root /tmp/imagenet_subset \
    --batch-size 32 \
    --epochs 5
```

## 📈 Expected Results

With proper training on ImageNet:

- **Target**: 75% top-1 accuracy (minimum for assignment)
- **Stretch Goal**: 78% top-1 accuracy
- **Training Time**: ~90 epochs, ~7-10 days on single GPU

## 🔍 Monitoring Training

### View Logs

```bash
# Training logs
cat logs/train_*.log

# Latest checkpoint info
ls -lh checkpoints/resnet50_imagenet/
```

### TensorBoard

```bash
tensorboard --logdir=./logs
# Open browser to http://localhost:6006
```

## 💾 Checkpoints

Checkpoints are saved in `checkpoints/resnet50_imagenet/`:

- `checkpoint.pth`: Latest checkpoint
- `best_model.pth`: Best validation accuracy model
- `checkpoint_epoch_N.pth`: Periodic checkpoints

## 🛠️ Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size in config.py
batch_size = 128  # or 64
```

### Slow Training

```python
# Increase number of workers
num_workers = 16

# Enable mixed precision (already enabled by default)
mixed_precision = True
```

### Data Loading Issues

```python
# Reduce workers if having issues
num_workers = 4

# Disable pin_memory
pin_memory = False
```

## 📝 Next Steps

1. ✅ Test on small dataset
2. ✅ Set up EC2 instance
3. ✅ Download ImageNet dataset
4. ✅ Start training
5. ⏳ Monitor training progress
6. ⏳ Achieve 75%+ accuracy
7. ⏳ Create HuggingFace app
8. ⏳ Record demo video

## 📚 References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [PyTorch ResNet Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- [DawnBench](https://dawn.cs.stanford.edu/benchmark/)
- [Fast.ai Imagenet Training](https://github.com/fastai/imagenet-fast)

## 📄 License

This project is for educational purposes as part of the ERA (Extensive Receptive Attention) course.

## 👥 Authors

Your Name(s) - Email(s)

---

**Note**: This is a baseline implementation. You may need to adjust hyperparameters, add more data augmentation, or implement additional techniques to reach the target accuracy.

