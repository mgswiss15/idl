import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        residual += out 
        out = self.relu(residual)
        
        return out


class PathMNISTResNet(nn.Module):
    def __init__(self, num_classes: int = 9, in_channels: int = 3) -> None:
        super().__init__()
        
        self.in_channels = 32
        self.conv1 = nn.Conv2d(
            in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(out_channels=32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=128, num_blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for s in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride=stride))
            self.in_channels = out_channels
            
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  
        
        x = self.avgpool(x) 
        x = torch.flatten(x, 1)
        x = self.fc(x)      
        
        return x