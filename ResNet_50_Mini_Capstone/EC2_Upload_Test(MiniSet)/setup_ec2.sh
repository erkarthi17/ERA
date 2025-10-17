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
sudo apt-get install -y python3-pip python3-venv python3-full git

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

# Create necessary directories in the parent folder
echo "Creating directories..."
mkdir -p ../checkpoints
mkdir -p ../logs
mkdir -p ../data

# (Optional) You might want to add a specific test for EC2_Upload_Test(MiniSet) if available.
# For now, we'll remove the general test_setup.py as it might not be relevant.
# python test_setup.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Ensure your sample data is in ./imagenette2 or a similar path within this directory."
echo "2. Run the training script: python train_hf.py --data-root ./imagenette2 --epochs 2"
echo ""
echo "To activate the virtual environment in future sessions (from this directory):"
echo "  source venv/bin/activate"
echo ""
