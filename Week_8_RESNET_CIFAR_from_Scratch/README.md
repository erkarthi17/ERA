# ResNet CIFAR-100 Classification Project

A comprehensive deep learning project implementing a custom ResNet architecture for CIFAR-100 image classification. This project demonstrates training a ResNet model from scratch with detailed logging, analysis, and deployment capabilities.

## 🎯 Project Overview

This project implements a custom ResNet44V2 architecture to classify images from the CIFAR-100 dataset. The model achieves **74.86% validation accuracy** through careful architecture design and training optimization.

### Key Features
- ✅ Custom ResNet44V2 implementation from scratch
- ✅ Comprehensive training logs and analysis
- ✅ Model checkpointing and best model saving
- ✅ Gradio web interface for inference
- ✅ Hugging Face integration for model deployment
- ✅ Detailed performance analysis and recommendations
- ✅ CPU-optimized training configuration

## 📊 Performance Results

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **74.86%** |
| **Final Training Accuracy** | 99.65% |
| **Final Validation Accuracy** | 74.70% |
| **Model Parameters** | ~2.7M |
| **Training Epochs** | 100 |

> 📈 **Note**: The model shows some overfitting (25% gap between train/val accuracy). See recommendations for improvement.

## 🏗️ Architecture

### ResNet44V2 Design
- **4-stage ResNet** with BasicBlockV2
- **Stage 1**: 32→32 channels, 7 blocks, stride 1
- **Stage 2**: 32→64 channels, 7 blocks, stride 2  
- **Stage 3**: 64→128 channels, 7 blocks, stride 2
- **Stage 4**: 128→256 channels, 3 blocks, stride 2
- **Final Layer**: Global average pooling + dropout(0.1) + FC layer

### Key Components
- **BatchNorm2d** with ReLU activations
- **Residual connections** for gradient flow
- **Progressive channel expansion** (32→64→128→256)
- **Optimized for CPU training** with balanced performance

## 📁 Project Structure

```
Week_8_RESNET_CIFAR_From_Scratch/
├── 📄 README.md                    # This file
├── 🔧 config.py                    # Training configuration
├── 🏗️ model.py                     # ResNet architecture definitions
├── 🚀 train.py                     # Main training script
├── 🛠️ utils.py                     # Utility functions
├── 📊 logs_RESNET.md               # Comprehensive training logs
├── 📋 requirements.txt             # Python dependencies
├── 📂 data/
│   ├── transforms.py               # Data augmentation pipeline
│   └── cifar-100-python/           # CIFAR-100 dataset
├── 📂 checkpoints/
│   └── best_model.pth              # Best trained model
├── 📂 huggingface_app/
│   ├── app.py                      # Gradio web interface
│   └── assets/                     # UI assets
└── 📂 test_data_setup/
    ├── get_test_images.py          # Test image extraction
    └── cifar100_test_images/       # Sample test images
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Week_8_RESNET_CIFAR_From_Scratch

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
# Start training (will download CIFAR-100 automatically)
python train.py
```

### 3. Inference

```bash
# Launch Gradio web interface
python huggingface_app/app.py
```

## ⚙️ Configuration

### Training Parameters (`config.py`)
```python
config = {
    "batch_size": 128,        # Optimized for CPU
    "epochs": 100,           # Training epochs
    "lr": 0.1,               # Initial learning rate
    "momentum": 0.9,         # SGD momentum
    "weight_decay": 5e-4,    # L2 regularization
    "log_file": "logs.md",   # Training logs
    "checkpoint_path": "checkpoints/best_model.pth"
}
```

### Data Augmentation
- **Training**: RandomCrop(32, padding=4) + RandomHorizontalFlip()
- **Validation**: Normalization only
- **Normalization**: CIFAR-100 specific mean/std values

## 📈 Training Details

### Optimizer & Scheduling
- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.1 initial with cosine annealing
- **Weight Decay**: 5e-4 for regularization
- **Batch Size**: 128 (CPU optimized)

### Training Progress
- **Real-time monitoring** with progress bars
- **Automatic checkpointing** of best model
- **Comprehensive logging** to markdown files
- **Training time estimation** and ETA display

## 🔍 Analysis & Insights

### Key Findings
1. **Strong Performance**: 74.86% validation accuracy on CIFAR-100
2. **Overfitting Concern**: 25% gap between training and validation accuracy
3. **Effective Architecture**: ResNet44V2 shows good capacity for the task
4. **Convergence Pattern**: Model converges around epoch 90-95

### Recommendations
1. **Early Stopping**: Stop training around epoch 80-85
2. **Stronger Regularization**: Increase dropout to 0.2-0.3
3. **Enhanced Augmentation**: Add Cutout, Mixup, or CutMix
4. **Learning Rate**: Reduce initial LR to 0.05

## 🌐 Web Interface

The project includes a Gradio web interface for easy model testing:

- **Upload images** for classification
- **Real-time predictions** with confidence scores
- **Top-5 predictions** display
- **User-friendly interface**

Launch with: `python huggingface_app/app.py`

## 📊 Monitoring & Logging

### Comprehensive Logs (`logs_RESNET.md`)
- ✅ **Model configuration** and architecture details
- ✅ **Training progress** with epoch-by-epoch metrics
- ✅ **Performance analysis** and comparisons
- ✅ **Recommendations** for future improvements
- ✅ **Quick reference** tables and summaries

### Training Metrics Tracked
- Training/Validation accuracy
- Loss values
- Learning rate schedule
- Best model checkpoints
- Training time and ETA

## 🛠️ Development

### Adding New Architectures
1. Define new model in `model.py`
2. Update training script to use new model
3. Adjust hyperparameters in `config.py`
4. Run training and analyze results

### Customizing Data Augmentation
Modify `data/transforms.py` to add new augmentation techniques:
```python
# Example: Add color jittering
transforms.ColorJitter(brightness=0.2, contrast=0.2)
```

## 📦 Dependencies

### Core Requirements
- `torch` - PyTorch framework
- `torchvision` - Computer vision utilities
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `tqdm` - Progress bars
- `gradio` - Web interface
- `huggingface_hub` - Model deployment
- `Pillow` - Image processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is part of the ERA (Effective ResNet Architecture) learning program.