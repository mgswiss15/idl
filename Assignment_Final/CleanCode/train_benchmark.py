import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

# =====================================================================
# 1. DATA PIPELINE MODULE
# =====================================================================
def get_pathmnist_loaders(batch_size=128):
    """
    Downloads and prepares data loaders for PathMNIST.
    PathMNIST consists of 28x28 RGB images across 9 distinct tissue classes.
    """
    info = INFO['pathmnist']
    
    # Transform scales [0, 255] uint8 images to [0.0, 1.0] and normalizes them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    DataClass = getattr(medmnist, info['python_class'])
    
    train_dataset = DataClass(split='train', transform=transform, download=True)
    val_dataset = DataClass(split='val', transform=transform, download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# =====================================================================
# 2. ARCHITECTURE MODULE (Streamlined ResNet)
# =====================================================================
class ResBlock(nn.Module):
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
    def __init__(self, num_classes=9):
        super().__init__()
        # Input stem optimized for small 28x28 images (no initial MaxPool)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)

        self.resblock1 = ResBlock(channels=32, identity_shortcut=True)  # Keeps 32 channels, 28x28
        self.resblock2 = ResBlock(channels=32, identity_shortcut=False) # Shifts to 64 channels, 14x14

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x


# =====================================================================
# 3. ENGINE MODULE (Trainer Lifecycle)
# =====================================================================
class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            # MedMNIST labels are [Batch, 1], CrossEntropy expects 1D target [Batch]
            labels = labels.squeeze().long() 
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / total, (correct / total) * 100

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.squeeze().long()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return running_loss / total, (correct / total) * 100

    def fit(self, train_loader, val_loader, epochs):
        print("\n Starting Training Routine...")
        print("-" * 55)
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        print("-" * 55)
        print("Training Benchmark Complete!")


# =====================================================================
# 4. MAIN ORCHESTRATION ENTRYPOINT
# =====================================================================
def main():
    # Set runtime device context
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")
    
    # Core Hyperparameters
    NUM_CLASSES = 9
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 5  # Set to 5 epochs for a quick verification run

    # Initialize data loaders
    train_loader, val_loader = get_pathmnist_loaders(batch_size=BATCH_SIZE)

    # Initialize components
    model = ResNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Execute training sequence
    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

if __name__ == "__main__":
    main()