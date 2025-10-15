# ResNet-50 ImageNet Training with Hugging Face Datasets

This guide shows how to use Hugging Face datasets to train ResNet-50 on ImageNet instead of using local ImageNet files.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install datasets huggingface_hub
```

### 2. Authenticate with Hugging Face

ImageNet is a gated dataset, so you need to be logged in:

```bash
huggingface-cli login
```

You'll need a Hugging Face account with access to ImageNet. If you don't have access, you can request it at: https://huggingface.co/datasets/imagenet-1k

### 3. Test the Setup

```bash
python test_hf_setup.py
```

### 4. Train with Subsets (for testing)

```bash
# Quick test with small subsets
python train_hf.py --train-subset 1000 --val-subset 500 --epochs 1 --batch-size 32

# Medium test
python train_hf.py --train-subset 10000 --val-subset 2000 --epochs 5 --batch-size 64
```

### 5. Full Training

```bash
# Full ImageNet training (90 epochs)
python train_hf.py --epochs 90 --batch-size 256
```

## 📁 New Files

- `data_hf.py` - Hugging Face data loader (replaces `data.py`)
- `train_hf.py` - Training script using Hugging Face datasets
- `test_hf_setup.py` - Setup verification script
- `README_HUGGINGFACE.md` - This guide

## 🔧 Key Features

### Subset Training
Perfect for testing and experimentation:
```bash
--train-subset 1000    # Use only 1000 training samples
--val-subset 500       # Use only 500 validation samples
```

### Automatic Download
The dataset is automatically downloaded and cached on first use.

### Drop-in Replacement
The Hugging Face data loader is designed to be a drop-in replacement for the original `data.py`.

## 📊 Benefits of Hugging Face Datasets

1. **No Local Storage**: No need to download and organize 150GB+ ImageNet files
2. **Automatic Caching**: Datasets are cached locally after first download
3. **Easy Subsetting**: Test with small subsets before full training
4. **Version Control**: Always get the latest, clean version of ImageNet
5. **Cloud Integration**: Works seamlessly in cloud environments

## 🛠️ Configuration

The training uses the same configuration as the original setup:

- **Model**: ResNet-50 (25.6M parameters)
- **Input Size**: 224x224
- **Classes**: 1000 (ImageNet)
- **Optimizer**: SGD with momentum
- **Scheduler**: Step LR scheduler
- **Mixed Precision**: Enabled by default

## 📈 Training Examples

### Quick Test (5 minutes)
```bash
python train_hf.py --train-subset 500 --val-subset 100 --epochs 1 --batch-size 16
```

### Development Test (30 minutes)
```bash
python train_hf.py --train-subset 5000 --val-subset 1000 --epochs 5 --batch-size 64
```

### Production Training (several hours)
```bash
python train_hf.py --epochs 90 --batch-size 256 --lr 0.1
```

## 🔍 Troubleshooting

### Authentication Issues
```bash
# Re-authenticate
huggingface-cli logout
huggingface-cli login
```

### Network Issues
- Ensure stable internet connection
- ImageNet is ~150GB, first download takes time
- Subsequent runs use cached data

### Memory Issues
- Reduce batch size: `--batch-size 32`
- Use smaller subsets: `--train-subset 1000`

### Dataset Access
- Ensure your Hugging Face account has ImageNet access
   - Request access at: https://huggingface.co/datasets/ILSVRC/imagenet-1k

## 📋 Comparison with Original Setup

| Feature | Original (`data.py`) | Hugging Face (`data_hf.py`) |
|---------|---------------------|----------------------------|
| Setup | Manual download | Automatic download |
| Storage | ~150GB local | Cached locally |
| Subsetting | Manual | Built-in |
| Updates | Manual | Automatic |
| Cloud Ready | No | Yes |

## 🎯 Next Steps

1. **Test Setup**: Run `python test_hf_setup.py`
2. **Quick Training**: Start with small subsets
3. **Full Training**: Scale up to full ImageNet
4. **Custom Datasets**: Modify `data_hf.py` for other Hugging Face datasets

## 💡 Tips

- Start with small subsets to verify everything works
- Use `--epochs 1` for quick validation
- Monitor GPU memory usage with larger batch sizes
- Check logs in `./logs/` directory
- Save checkpoints regularly with `--save-every 10`
