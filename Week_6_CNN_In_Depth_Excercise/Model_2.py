# --------------------------------------------------------------------------------------------------
# Text Block

# What does this Model Comprises of?
# This model is an iterative version of Model_1.py, which was based on a basic CNN structure.
# This new version incorporates next-level features for improved efficiency and performance,
# including Batch Normalization, Dropout Regularization, and Global Average Pooling.

# What's the expected outcome of this Model as TARGET?
# With the introduction of Batch Normalization, Dropout Regularization, and Global Average Pooling,
# the model is expected to be more efficient, and the gap between training and test accuracy
# is significantly reduced.

# Results:
# Total number of parameters handled by this version of Model is: 5710 (a significant reduction from initial designs)
# Best Training Accuracy achieved by the Model is: 98.68%
# Best Test Accuracy achieved by the Model is: 98.95%

# What's the next Objective?
# To further enhance the model's robustness, the following key features will be addressed in the next version:
# 1. Increasing the model's capacity
# 2. Strategic placement of MaxPooling layers to reduce spatial dimensions
# 3. Implementing Image Augmentation
# 4. Introduction of Learning Rate scheduling

# Analysis:
# 1. Initially, the model's parameter count increased with the introduction of Batch Normalization 
# compared to the optimized Model_1.py.
# However, with the subsequent addition of Dropout Regularization and Global Average Pooling, the total parameters 
# were effectively reduced to 5.7K.
# 2. The training accuracy of this Model on Epoch 0 (95.95%) was significantly better than the previous model (82.86%).
# This improvement is attributed to Batch Normalization, which stabilizes activations within each layer.
# 3. The gap between training and testing accuracy has been reduced with the help of Dropout Regularization.
# 4. A final 1x1 vector was effectively achieved by introducing Global Average Pooling.
# 5. Due to the considerable reduction in the total number of parameters for this model, we might observe 
# a slight dip in overall accuracy (both training and testing).
# This concern will be addressed in the next version of the model, Model_3.py, by increasing capacity.
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
        self.dropout_value = 0.1

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=0, bias=False), # Output 26x26, RF=3
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0, bias=False), # Output 24x24, RF=5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value)
        )

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False), # Output 24x24, RF=5
        )
        self.pool1 = nn.MaxPool2d(2, 2) # Output 12x12, RF=6

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0, bias=False), # Output 10x10, RF=10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=0, bias=False), # Output 8x8, RF=14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value)
        )

        # OUTPUT BLOCK with GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8) # Output 1x1, RF=14
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False), # Output 1x1, RF=14
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.gap(x)
        x = self.convblock6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Function to prepare data loaders for Model_2
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