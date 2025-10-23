# EC2 Quick Start Guide

## 🚀 Quick Testing on EC2 (2-5 epochs)

### Step 1: Launch EC2 Instance

```bash
# Recommended for testing: p3.2xlarge (1 GPU, 8 vCPU, 61 GB RAM) but tested in g5* instance
# AMI: Deep Learning AMI (Ubuntu) - PyTorch 2.0+
```

### Step 2: SSH into EC2

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### Step 3: Setup Environment

```bash
# Clone your repository
git clone <your-repo-url>
cd ResNet_50_Mini_Capstone

# Run setup script
chmod +x setup_ec2.sh
./setup_ec2.sh

# Activate virtual environment
source venv/bin/activate
```

### Step 4: Download ImageNet (if not already done)

```bash
# Create data directory
sudo mkdir -p /data/imagenet
sudo chown -R $USER:$USER /data/imagenet

aws s3 sync s3://imagenet-dataset-karthick-kannan /data/imagenet

# Download ImageNet (this takes time!)
# Option 1: From official source
# Visit: https://www.image-net.org/download.php

# Option 2: From S3 (if you have it there)
# aws s3 sync s3://your-bucket/imagenet /data/imagenet
```

### Step 5: Quick Test (2 epochs)

```bash
# Option A: Using the test script (RECOMMENDED)
python test_training.py \
    --data-root /data/imagenet \
    --epochs 2 \
    --batch-size 64 \
    --lr 0.01

# Option B: Using train.py directly
python train.py \
    --data-root /data/imagenet \
    --epochs 2 \
    --batch-size 64 \
    --lr 0.01
```

### Step 6: Monitor Training

```bash
# In another terminal, SSH into EC2 and monitor:
ssh -i your-key.pem ubuntu@your-ec2-ip

# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor logs
tail -f logs/train_*.log

# Or use TensorBoard
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
# Then access: http://your-ec2-ip:6006
```

## 📊 Expected Test Results

With 2 epochs on ImageNet:
- **Epoch 1**: Training loss should decrease from ~6.9 to ~5.5
- **Epoch 2**: Training loss should decrease from ~5.5 to ~4.8
- **Top-1 Accuracy**: Should reach 1-3% (random is 0.1%)
- **Time**: ~2-4 hours on p3.2xlarge

## 🎯 Different Test Scenarios

### Scenario 1: Quick Smoke Test (1 epoch)
```bash
python train.py --data-root /data/imagenet --epochs 1 --batch-size 32
```

### Scenario 2: Medium Test (5 epochs)
```bash
python train.py --data-root /data/imagenet --epochs 5 --batch-size 128
```

### Scenario 3: Realistic Test (10 epochs)
```bash
python train.py --data-root /data/imagenet --epochs 10 --batch-size 256
```

## 💡 Tips for Testing

### 1. Use Smaller Batch Size for Testing
```bash
# Testing: batch-size 64
# Full training: batch-size 256
```

### 2. Reduce Workers for Testing
```python
# Edit config.py
num_workers = 4  # Instead of 8
```

### 3. Test on Subset First (Optional)
```bash
# Create a 10-class subset for ultra-fast testing
mkdir -p /tmp/imagenet_subset/{train,val}
# Copy 10 random classes from ImageNet
# Then test:
python train.py --data-root /tmp/imagenet_subset --epochs 1
```

### 4. Monitor System Resources
```bash
# CPU and Memory
htop

# GPU
nvidia-smi -l 1

# Disk I/O
iostat -x 1
```

## 🔧 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python train.py --batch-size 32 --epochs 2 ...
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Should be 90-100% for GPU
# If low, check data loading (CPU bottleneck)
```

### Data Loading Issues
```bash
# Reduce workers
# Edit config.py: num_workers = 2
```

## ✅ After Successful Test

Once your 2-epoch test completes successfully:

1. **Check logs**: `cat logs/train_*.log`
2. **Check checkpoints**: `ls -lh checkpoints/resnet50_imagenet/`
3. **Verify accuracy**: Should see improvement from epoch 1 to 2

Then proceed with full training:

```bash
# Full training (90 epochs)
python train.py \
    --data-root /data/imagenet \
    --epochs 90 \
    --batch-size 256 \
    --lr 0.1
```

## 📝 Quick Commands Cheat Sheet

```bash
# Start training in background
nohup python train.py --data-root /data/imagenet --epochs 2 > training.log 2>&1 &

# Check if training is running
ps aux | grep train.py

# Monitor logs
tail -f training.log

# Stop training
pkill -f train.py

# Check GPU
nvidia-smi

# Check disk space
df -h

# Check memory
free -h
```

## 🎓 Next Steps

After successful 2-epoch test:
1. ✅ Verify training loop works
2. ✅ Check checkpoint saving
3. ✅ Verify validation works
4. ✅ Check logs are being written
5. 🚀 Start full 90-epoch training!

---

**Estimated Costs for Testing:**
- 2 epochs: ~$2-5 (2-4 hours on p3.2xlarge)
- 5 epochs: ~$5-10 (5-8 hours)
- Full 90 epochs: ~$150-250 (7-10 days)

