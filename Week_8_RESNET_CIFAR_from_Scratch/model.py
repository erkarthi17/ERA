import torch
import torch.nn as nn

class BasicBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        shortcut = self.shortcut(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        return out + shortcut

# Optimized ResNet for CPU training - balanced performance and speed
class ResNet44V2(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # Moderate initial channels for good capacity without being too heavy
        self.init_conv = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.init_bn = nn.BatchNorm2d(32)
        self.init_relu = nn.ReLU(inplace=True)
        
        # Balanced architecture: 4 stages with moderate depth
        self.stage1 = self._make_layer(32, 32, 7, 1)   # 7 blocks in stage1
        self.stage2 = self._make_layer(32, 64, 7, 2)   # 7 blocks in stage2
        self.stage3 = self._make_layer(64, 128, 7, 2)  # 7 blocks in stage3
        self.stage4 = self._make_layer(128, 256, 3, 2) # 3 blocks in stage4
        
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlockV2(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlockV2(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.init_bn(out)
        out = self.init_relu(out)
        
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Even lighter version for very fast CPU training
class ResNet32V2_Optimized(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # Slightly increased channels from original but not too heavy
        self.init_conv = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.init_bn = nn.BatchNorm2d(32)
        self.init_relu = nn.ReLU(inplace=True)
        
        # 4 stages with moderate depth
        self.stage1 = self._make_layer(32, 32, 5, 1)   # 5 blocks
        self.stage2 = self._make_layer(32, 64, 5, 2)   # 5 blocks
        self.stage3 = self._make_layer(64, 128, 5, 2)  # 5 blocks
        self.stage4 = self._make_layer(128, 256, 2, 2) # 2 blocks
        
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlockV2(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlockV2(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.init_bn(out)
        out = self.init_relu(out)
        
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Keep the original for comparison
class ResNet32V2(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.stage1 = self._make_layer(16, 16, 5, 1)
        self.stage2 = self._make_layer(16, 32, 5, 2)
        self.stage3 = self._make_layer(32, 64, 5, 2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlockV2(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlockV2(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out