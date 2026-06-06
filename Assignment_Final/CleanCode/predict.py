import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from data import get_loaders
import models

@torch.no_grad()
def main(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Unbiased Production Audit on: {device}")

    # 1. Isolate the Test Stream (ignoring train/val entirely)
    _, _, test_loader = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

    model_class = getattr(models, config["MODEL"])
    model = model_class(
        in_channels=config["CHANNELS"], 
        num_classes=config["NUM_CLASSES"],
        **config
    ).to(device)

    model_weights_path = "../checkpoints/"+config['DATA']+"_model.pth"
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print(f"Successfully loaded {model_weights_path} weights.")

    model.eval()
    all_preds = []
    all_targets = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.view(-1).cpu().numpy())

    target_names = [f"Class {i}" for i in range(config["NUM_CLASSES"])]
    
    print("\n --- FINAL COMPLIANCE REPORT --- ")
    report_dict = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0, output_dict=True)
    print(report_dict)

    # 6. Plot Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(config["NUM_CLASSES"])))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    
    plt.title(f"Deployment Confusion Matrix (Classes: {config['NUM_CLASSES']})")
    plt.tight_layout()
    plt.savefig("./results/" + config["DATA"] + "_confusion.png", dpi=300)
    print(f"Analysis visualization saved as {config["DATA"]}_confusion.png\n")

    return report_dict

if __name__ == "__main__":
    main()