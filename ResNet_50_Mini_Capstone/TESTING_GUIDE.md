# Testing Guide for ResNet-50 Training

This guide shows you how to test the ResNet-50 training pipeline locally using smaller datasets.

## Quick Start

### Option 1: Use CIFAR-10 (Recommended for Local Testing)

CIFAR-10 is automatically downloaded and converted to ImageNet-like structure:

```bash
cd ResNet_50_Mini_Capstone
python test_training.py --dataset cifar10 --epochs 2 --batch-size 32
```

**What this does:**
- Downloads CIFAR-10 automatically (~160MB)
- Converts it to ImageNet directory structure
- Uses 10 classes instead of 1000
- Limits dataset size for faster testing (1000 train, 200 val per class)

### Option 2: Use Mini ImageNet Structure

Creates empty ImageNet-like directory structure:

```bash
python test_training.py --dataset mini-imagenet --epochs 2 --batch-size 32
```

**What this does:**
- Creates ImageNet-like directory structure
- No actual images (for testing directory structure)
- You can manually add images later

### Option 3: Use Your Own ImageNet Subset

If you have a subset of ImageNet:

```bash
python test_training.py --dataset custom --data-root /path/to/your/imagenet/subset
```

## Dataset Options

| Option | Size | Download | Classes | Use Case |
|--------|------|----------|---------|----------|
| `cifar10` | ~160MB | Auto | 10 | Quick local testing |
| `mini-imagenet` | ~1KB | None | 10 | Structure testing |
| `custom` | Variable | Manual | 1000 | Your own subset |

## Command Line Arguments

- `--dataset`: Choose dataset type (`cifar10`, `mini-imagenet`, `custom`)
- `--epochs`: Number of training epochs (default: 2)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.01)
- `--data-root`: Custom data path (required for `custom` dataset)

## Example Commands

```bash
# Quick test with CIFAR-10
python test_training.py --dataset cifar10

# Test with custom settings
python test_training.py --dataset cifar10 --epochs 5 --batch-size 16 --lr 0.001

# Test with your own data
python test_training.py --dataset custom --data-root /data/imagenet --epochs 1
```

## Troubleshooting

1. **Out of memory**: Reduce batch size (`--batch-size 16` or `--batch-size 8`)
2. **Slow training**: Reduce epochs (`--epochs 1`) or use smaller dataset
3. **Permission errors**: Check write permissions for temp directories
4. **Download fails**: Check internet connection for CIFAR-10 download

## Expected Output

You should see:
- Dataset creation messages
- Training progress with loss and accuracy
- Model checkpointing
- Final results

## Next Steps

Once local testing works:
1. Use the same script on EC2 with full ImageNet
2. Adjust hyperparameters based on results
3. Scale up to full training runs
