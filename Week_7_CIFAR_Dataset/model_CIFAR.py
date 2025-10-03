from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# CIFAR-10 dataset mean and std deviation
# Mean: (0.4914, 0.4822, 0.4465)
# Std Dev: (0.2471, 0.2435, 0.2616)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout_value = 0.1

        # Prep Layer - C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False), # Increased from 16 to 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # Output: 32x32x32 | RF: 3

        # Conv Block 1 - C2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), # Increased from 16 to 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # Output: 32x32x64 | RF: 5

        # Transition Block 1 - C3 (Downsample)
        self.transition1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False), # Input from 16 to 64, output 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # Output: 16x16x64 | RF: 9

        # Conv Block 2 - C4 (Dilated)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2, bias=False), # Dilated convolution, input/output 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # Output: 16x16x64 | RF: 17

        # Transition Block 2 - C5 (Downsample)
        self.transition2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # Increased from 32 to 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # Output: 8x8x128 | RF: 25

        # Depthwise Separable Conv Block - C6 (Depthwise + Pointwise)
        self.depthwise_separable = nn.Sequential(
            # Depthwise
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False), # Increased from 64 to 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            # Pointwise
            nn.Conv2d(128, 128, kernel_size=1, bias=False), # Increased from 64 to 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # Output: 8x8x128 | RF: 29

        # Output Block - C7 (1x1 to reduce channels to 10 for GAP)
        self.output_conv = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=1, bias=False), # Input from 64 to 128
            # No BatchNorm or ReLU here, as it's directly before GAP and log_softmax
        ) # Output: 8x8x10 | RF: 29

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1) # Output: 1x1x10 | RF: 57

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transition1(x)
        x = self.convblock3(x)
        x = self.transition2(x)
        x = self.depthwise_separable(x)
        x = self.output_conv(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Albumentations transforms
class AlbumentationTransforms:
    def __init__(self, train=True):
        self.train = train
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=1,
                            min_holes=1, min_height=16, min_width=16,
                            fill_value=mean, mask_fill_value=None, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        self.test_transform = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    def __call__(self, img):
        img = np.array(img) # Convert PIL Image to NumPy array
        if self.train:
            return self.train_transform(image=img)['image']
        return self.test_transform(image=img)['image']

# Function to prepare data loaders for CIFAR_Model
def get_dataloaders(cuda_available, batch_size=512): # Increased batch size for CIFAR
    train_transforms = AlbumentationTransforms(train=True)
    test_transforms = AlbumentationTransforms(train=False)

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda_available else dict(shuffle=True, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    return train_loader, test_loader