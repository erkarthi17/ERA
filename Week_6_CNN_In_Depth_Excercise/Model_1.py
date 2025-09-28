# --------------------------------------------------------------------------------------------------
# Text Block

# What does this Model Comprises of?
# This model started as a basic skeleton (without advanced concepts such as 
# Batch Normalization, Regularization, GAP, Image augmentation & Learning Rate allocation)
# and has been optimized for efficiency and performance.

# What's the expected outcome of this Model as TARGET?
# This model is a significantly lighter version compared to its initial design,
# achieving high accuracy after architectural optimizations.
# Key features missing in this model will be handled in Model_2.py, Model_3.py accordingly.

# Results:
# Total number of parameters handled by this version of Model is: 3668
# Best Training Accuracy achieved by the Model is: 97.50%
# Best Test Accuracy achieved by the Model is: 97.36%

# What's the next Objective?
# To make this model slightly better and handle following three features:
# 1. Batch Normalization
# 2. Regularization
# 3. Global Average Pooling

# Analysis:
# 1. I've started to design the model from the base with a huge number of parameters (initially 6.3M)
# and worked on redefining the model architecture by introducing an appropriate number
# of in/out channels in the Convolutional Block and with the help of the Transformation Block,
# was able to bring the total number of parameters to 10.7K.
# 2. I could see a significant reduction in the execution time of model prediction completion
# post changing the total number of parameters. So, based on this, I clearly understood that
# having the right level of CNN architecture is essential to have a lighter model which predicts
# quickly and reduces the burden on GPU/CPU.
# --------------------------------------------------------------------------------------------------

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Model Definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=0, bias=False), # Output 26x26, RF=3
            nn.ReLU()
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=0, bias=False), # Output 24x24, RF=5
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2) # Output 12x12, RF=6
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=0, bias=False), # Output 10x10, RF=10
            nn.ReLU()
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False), # Output 10x10, RF=10
            nn.ReLU()
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=10) # Output 1x1, RF=10
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Function to prepare data loaders for Model_1
def get_dataloaders(cuda_available, batch_size=128):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda_available else dict(shuffle=True, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    return train_loader, test_loader