import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """ResBlock with 3x3 convolutions - He et al.'s original ResNet paper."""
    def __init__(self, channels, identity_shortcut=True):
        super().__init__()
        if identity_shortcut:
            out_channels = channels
            stride = 1
            self.shortcut = nn.Identity()
        else:
            out_channels = channels * 2
            stride = 2
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.conv1 = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet with 2 resblocks - one identify, one expanding."""
    def __init__(self, in_channels, num_classes):
        super().__init__()      
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)

        self.resblock1 = ResBlock(channels=32, identity_shortcut=True)
        self.resblock2 = ResBlock(channels=32, identity_shortcut=False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x