#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting EC2 setup for ResNet-50 ImageNet S3 training..."

# Ensure we are in the correct directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

echo "Current directory: $(pwd)"

# 1. Update and install system dependencies
echo "Updating system packages and installing git, tmux, htop..."
sudo apt-get update -y
sudo apt-get install -y git tmux htop build-essential

# 2. Install Miniconda (if not already installed, common on DL AMIs)
if ! command -v conda &> /dev/null
then
    echo "Conda not found. Installing Miniconda..."
    MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER -O /tmp/$MINICONDA_INSTALLER
    bash /tmp/$MINICONDA_INSTALLER -b -p $HOME/miniconda3
    rm /tmp/$MINICONDA_INSTALLER
    export PATH="$HOME/miniconda3/bin:$PATH"
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    echo "Miniconda installed."
else
    echo "Conda already installed."
fi

# 3. Create and activate a Python virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
echo "Installing Python dependencies from parent requirements.txt..."
# Point to the requirements.txt in the parent ResNet_50_Mini_Capstone directory
pip install -r ../requirements.txt

echo "Environment setup complete!"
echo "To activate the environment in new sessions, run: source venv/bin/activate"
echo "You can now run your training script: python train_s3.py --s3-bucket imagenet-dataset-karthick-kannan ..."