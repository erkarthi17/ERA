# --------------------------------------------------------------------------------------------------
# Text Block

# What does this Model Comprises of?
# This model is an ultra-efficient iteration, building upon Model_2.py. It integrates strategic
# channel reduction, optimized architecture design, advanced Image Augmentation, and precise
# Learning Rate Scheduling to achieve high performance with minimal parameters.

# What's the expected outcome of this Model as TARGET?
# This model is specifically designed to meet the following criteria:
# 1. Test Accuracy: Consistently >= 99% (targeting 99.4%+)
# 2. Parameters: Under 8000 trainable parameters (achieved: 3,928)
# 3. Epochs: <= 15 epochs for complete training
# 4. Training Stability: Smooth, ascending accuracy curve

# Results:
# Total number of parameters handled by this version of Model is: 3,928
# Best Training Accuracy achieved by the Model is: 98.04%
# Best Test Accuracy achieved by the Model is: 98.55%

# What's the next Objective?
# This model successfully demonstrates achieving high accuracy with minimal parameters.
# Future objectives could explore even more aggressive parameter reduction or
# application to more complex datasets.
# Fine Tuning is required to bring the model acheive the target and maintain >99.4% consistently

# Analysis:
# 1. Parameter Count: The architecture has been meticulously designed to have 3,928 trainable parameters,
# which is well under the 8K target while maintaining sufficient capacity for high accuracy.
# 2. 7 Convolutional Layers: The model incorporates 7 distinct convolutional steps with strategic
# channel progression (8->10->8->10->12->10->12->16->10 output from FC) to ensure efficient feature extraction.
# 3. Receptive Field: The receptive field of this model is 24x24, allowing it to capture global context
# effectively for a 28x28 input.
#    RF Calculation Details:
#    - Input: RF = 1, Jump = 1
#    - convblock1 (k=3, s=1): RF = 3, Jump = 1
#    - convblock2 (k=3, s=1): RF = 5, Jump = 1
#    - convblock3_1x1 (k=1, s=1): RF = 5, Jump = 1
#    - pool1 (k=2, s=2): RF = 6, Jump = 2
#    - convblock4 (k=3, s=1): RF = 10, Jump = 2
#    - convblock5 (k=3, s=1): RF = 14, Jump = 2
#    - convblock6_1x1 (k=1, s=1): RF = 14, Jump = 2
#    - pool2 (k=2, s=2): RF = 16, Jump = 4
#    - convblock7 (k=3, s=1): RF = 24, Jump = 4
#    - gap (k=2, s=1 on 2x2 output): RF = 24
#    - fc (k=1, s=1): RF = 24
# 4. Optimized Learning: Adam optimizer with OneCycleLR for fast convergence and stability.
# 5. Data Augmentation: ColorJitter and RandomRotation for robust generalization.
# --------------------------------------------------------------------------------------------------

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Model Definition - Ultra-efficient architecture under 5K parameters (Target: 3,928 parameters)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout_value = 0.1

        # Layer 1 (Input Block)
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=0, bias=False),  # 28x28 -> 26x26 (RF=3)
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_value)
        )

        # Layer 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8, 10, 3, padding=0, bias=False), # 26x26 -> 24x24 (RF=5)
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        )

        # Layer 3 (Transition Block: 1x1 Conv)
        self.convblock3_1x1 = nn.Sequential(
            nn.Conv2d(10, 8, 1, padding=0, bias=False), # 24x24 -> 24x24 (RF=5)
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_value)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 24x24 -> 12x12 (RF=6)

        # Layer 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(8, 10, 3, padding=0, bias=False), # 12x12 -> 10x10 (RF=10)
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        )

        # Layer 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(10, 12, 3, padding=0, bias=False), # 10x10 -> 8x8 (RF=14)
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(self.dropout_value)
        )
        
        # Layer 6 (Transition 1x1 before final pooling)
        self.convblock6_1x1 = nn.Sequential(
            nn.Conv2d(12, 10, 1, padding=0, bias=False), # 8x8 -> 8x8 (RF=14)
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        )
        self.pool2 = nn.MaxPool2d(2, 2) # 8x8 -> 4x4 (RF=16)

        # Layer 7 (Output block before GAP)
        self.convblock7 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=0, bias=False), # 4x4 -> 2x2 (RF=24)
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        )

        # Global Average Pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2) # 2x2 -> 1x1
        )

        # Output Layer (1x1 Conv acting as FC)
        self.fc = nn.Conv2d(10, 10, 1, bias=False)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3_1x1(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6_1x1(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Function to prepare data loaders for Model_3 (with augmentation)
def get_dataloaders(cuda_available, batch_size=256):
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.RandomRotation(degrees=(-7.0, 7.0), fill=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda_available else dict(shuffle=True, batch_size=128)
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    return train_loader, test_loader