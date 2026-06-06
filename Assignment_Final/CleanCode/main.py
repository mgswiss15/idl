import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
from model import ResNet
from train import Trainer

def main():   
    NUM_CLASSES = 9
    CHANNELS = 3
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    train_loader, val_loader, _ = get_loaders(batch_size=BATCH_SIZE)

    model = ResNet(in_channels=CHANNELS, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

if __name__ == "__main__":
    main()