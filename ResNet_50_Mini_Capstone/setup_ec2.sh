#!/bin/bash
# EC2 Setup Script for ResNet-50 ImageNet Training
# Run this script on your EC2 instance after launching

set -e  # Exit on error

echo "=========================================="
echo "EC2 Setup for ResNet-50 Training"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip if not already installed
echo "Installing Python dependencies..."
sudo apt-get install -y python3-pip python3-venv git

# Install NVIDIA drivers (if using GPU instance)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected!"
    nvidia-smi
else
    echo "No NVIDIA GPU detected. Using CPU (will be slow)."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (with CUDA support if GPU available)
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio
fi

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p data

# Test installation
echo "Testing installation..."
python test_setup.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download ImageNet dataset to /data/imagenet"
echo "2. Update config.py with correct data path"
echo "3. Run: python train.py --data-root /data/imagenet"
echo ""
echo "To activate the virtual environment in future sessions:"
echo "  source venv/bin/activate"
echo ""

