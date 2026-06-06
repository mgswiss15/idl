"""
Workings for MAI/IDL - prepare the PathMNIST dataset for student use by downloading, transforming, and saving it in a single .pt file.

Usage: Fast and dirty solution without any flexibility. 
Used for these two datasets:
1. medmnist.PathMNIST (RGB, 9 classes) to getnerate 'histology_data.pt'
2. medmnist.TissueMNIST (Grayscale, 9 classes) to generate 'dapi_data.pt' - see here for description and reason for DAPI name: https://bbbc.broadinstitute.org/BBBC051

MG 6/6/2026
"""
import torch
import medmnist
from torchvision import transforms

def generate_student_dataset():
    # 1. Download/Load raw data
    train_dataset = medmnist.TissueMNIST(split='train', download=True, root='../data')
    val_dataset = medmnist.TissueMNIST(split='val', download=True, root='../data')
    test_dataset = medmnist.TissueMNIST(split='test', download=True, root='../data')
    
    # 2. Transform to tensors and normalize right away
    # This turns images into float tensors of shape (3, 28, 28) with values ~ [-1.0, 1.0]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print("Converting training data to tensors...")
    train_imgs = torch.stack([transform(img) for img, _ in train_dataset])
    train_lbls = torch.tensor([label for _, label in train_dataset]) # Keeps original shape [N, 1]
    
    print("Converting validation data to tensors...")
    val_imgs = torch.stack([transform(img) for img, _ in val_dataset])
    val_lbls = torch.tensor([label for _, label in val_dataset])
    
    print("Converting test data to tensors...")
    test_imgs = torch.stack([transform(img) for img, _ in test_dataset])
    test_lbls = torch.tensor([label for _, label in test_dataset])

    # 3. Save into a single file dictionary
    payload = {
        'train_images': train_imgs,
        'train_labels': train_lbls,
        'val_images': val_imgs,
        'val_labels': val_lbls,
        'test_images': test_imgs,
        'test_labels': test_lbls
    }
    
    torch.save(payload, '../data/dapi_data.pt')
    print("File 'dapi_data.pt' successfully created and ready for distribution!")

if __name__ == "__main__":
    generate_student_dataset()