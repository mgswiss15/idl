import json

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
from model import ResNet
from train import Trainer

def main():   
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    train_loader, val_loader, test_loader = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

    model = ResNet(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])

    # Save the trained weights to a file artifact
    model_weights_path = "../checkpoints/"+config['DATA']+"_model.pth"
    torch.save(model.state_dict(), model_weights_path)
    print(f"Training complete. Weights safely exported to '{model_weights_path}'")


if __name__ == "__main__":
    main()
    