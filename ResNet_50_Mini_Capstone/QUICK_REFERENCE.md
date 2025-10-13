# Quick Reference: Limiting Training Epochs

## 🎯 Three Ways to Limit Epochs

### Method 1: Command Line (RECOMMENDED for EC2)

```bash
# Test with 2 epochs
python train.py --data-root /data/imagenet --epochs 2

# Test with 5 epochs
python train.py --data-root /data/imagenet --epochs 5

# Test with custom batch size and learning rate
python train.py \
    --data-root /data/imagenet \
    --epochs 3 \
    --batch-size 64 \
    --lr 0.01
```

### Method 2: Using Test Script

```bash
# Quick 2-epoch test (uses default settings)
python test_training.py --data-root /data/imagenet

# Custom test
python test_training.py \
    --data-root /data/imagenet \
    --epochs 5 \
    --batch-size 128 \
    --lr 0.01
```

### Method 3: Edit config.py

```python
# Edit config.py
config = Config(
    epochs=2,              # Change from 90 to 2
    batch_size=64,         # Smaller for testing
    learning_rate=0.01,    # Lower for testing
    # ... other params
)
```

## 📋 All Available Command Line Arguments

```bash
python train.py \
    --data-root /path/to/imagenet \    # REQUIRED: Path to ImageNet
    --epochs 90 \                       # Number of epochs (default: 90)
    --batch-size 256 \                  # Batch size (default: 256)
    --lr 0.1 \                          # Learning rate (default: 0.1)
    --resume /path/to/checkpoint.pth    # Resume from checkpoint
```

## 🧪 Testing Scenarios

### Scenario 1: Quick Smoke Test (1 hour)
```bash
python train.py --data-root /data/imagenet --epochs 1 --batch-size 32
# Purpose: Verify code works, no crashes
# Time: ~30-60 minutes
```

### Scenario 2: Short Test (2-4 hours)
```bash
python train.py --data-root /data/imagenet --epochs 2 --batch-size 64
# Purpose: Verify training loop, checkpointing, validation
# Time: ~2-4 hours
```

### Scenario 3: Medium Test (1 day)
```bash
python train.py --data-root /data/imagenet --epochs 10 --batch-size 128
# Purpose: Verify convergence, learning rate schedule
# Time: ~12-24 hours
```

### Scenario 4: Full Training (7-10 days)
```bash
python train.py --data-root /data/imagenet --epochs 90 --batch-size 256
# Purpose: Complete training to 75%+ accuracy
# Time: ~7-10 days
```

## 🔍 Monitoring During Test

```bash
# Terminal 1: Start training
python train.py --data-root /data/imagenet --epochs 2

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 3: Monitor logs
tail -f logs/train_*.log

# Terminal 4: TensorBoard
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
```

## 📊 What to Check During 2-Epoch Test

### ✅ Epoch 1:
- [ ] Training starts without errors
- [ ] Loss decreases (from ~6.9 to ~5.5)
- [ ] GPU utilization > 90%
- [ ] Checkpoint saved

### ✅ Epoch 2:
- [ ] Training continues from checkpoint
- [ ] Loss continues to decrease
- [ ] Validation accuracy improves
- [ ] Logs are being written

### ✅ After Test:
- [ ] Checkpoint files exist
- [ ] Best model saved
- [ ] Logs are complete
- [ ] No errors in logs

## 💰 Cost Estimates (p3.2xlarge @ ~$3/hour)

| Epochs | Batch Size | Time | Cost |
|--------|-----------|------|------|
| 1 | 32 | 30-60 min | $1-2 |
| 2 | 64 | 2-4 hours | $6-12 |
| 5 | 128 | 5-8 hours | $15-24 |
| 10 | 128 | 12-24 hours | $36-72 |
| 90 | 256 | 7-10 days | $500-720 |

## 🚨 Common Issues

### Issue: Out of Memory
```bash
# Solution: Reduce batch size
python train.py --batch-size 32 --epochs 2 ...
```

### Issue: Training too slow
```bash
# Check GPU utilization
nvidia-smi
# If < 90%, reduce num_workers in config.py
```

### Issue: Can't see progress
```bash
# Check logs
cat logs/train_*.log

# Or use TensorBoard
tensorboard --logdir=./logs
```

## 📝 Quick Commands

```bash
# Start training in background
nohup python train.py --data-root /data/imagenet --epochs 2 > train.log 2>&1 &

# Check if running
ps aux | grep train.py

# View logs
tail -f train.log

# Stop training
pkill -f train.py

# Check checkpoints
ls -lh checkpoints/resnet50_imagenet/
```

## 🎯 Recommended Testing Workflow

1. **First**: 1 epoch with batch-size 32 (verify it works)
2. **Second**: 2 epochs with batch-size 64 (verify checkpointing)
3. **Third**: 5 epochs with batch-size 128 (verify convergence)
4. **Finally**: Full 90 epochs with batch-size 256 (complete training)

---

**Remember**: Always test with 2 epochs before committing to full training!

