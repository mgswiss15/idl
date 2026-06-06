import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch

@torch.no_grad()
def evaluate_and_plot(model, val_loader, device, num_classes):
    """
    Runs evaluation on the validation set, prints a classification report,
    and displays/saves a confusion matrix.
    """
    model.eval()
    all_preds = []
    all_targets = []

    # 1. Gather all predictions and ground truths
    for images, labels in val_loader:
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        # Append to CPU lists (flatten targets if they are [B, 1])
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.view(-1).cpu().numpy())

    # 2. Generate the textual precision/recall report
    # Using explicit target names based on config's expected class count
    target_names = [f"Class {i}" for i in range(num_classes)]
    
    print("\n📊 --- PRODUCTION CLASSIFICATION REPORT --- 📊")
    # print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))

    # 3. Compute and plot the Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    
    plt.title(f"Confusion Matrix (Configured Classes: {num_classes})")
    plt.tight_layout()
    
    # Save image to disk so students can inspect it
    plt.savefig("confusion_matrix.png", dpi=300)
    print("💾 Confusion matrix visualization saved as 'confusion_matrix.png'")
    plt.show()