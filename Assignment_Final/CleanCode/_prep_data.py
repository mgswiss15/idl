"""
Workings for MAI/IDL - prepare the PathMNIST dataset for student use by downloading, transforming, and saving it in a single .pt file.

Usage: Fast and dirty solution without any flexibility. 
Original description of datasets - https://arxiv.org/pdf/2110.14795
Used for these datasets:
1. dataset_name="PathMNIST", output_name='histology' (RGB, 9 classes)
2. dataset_name="TissueMNIST", output_name='dapi' (Grayscale, 8 classes) - https://bbbc.broadinstitute.org/BBBC051
3. dataset_name="OCTMNIST", output_name='liver' (Grayscale, 4 classes) - https://pubmed.ncbi.nlm.nih.gov/36481607/
4. dataset_name="BloodMNIST", output_name='cells' (RGB, 8 classes)
5. dataset_name="OrganSMNIST", output_name='organs' (Grayscale, 11 classes)

MG 6/6/2026
"""
import torch
import torch.nn as nn
import medmnist
from torchvision import transforms

def generate_student_dataset(dataset_name="BloodMNIST", output_name='cells', size=64):
    # 1. Download/Load raw data
    dataset_class = getattr(medmnist, dataset_name)
    train_dataset = dataset_class(split='train', download=True, size=size, root='../data')
    val_dataset = dataset_class(split='val', download=True, size=size, root='../data')
    test_dataset = dataset_class(split='test', download=True, size=size, root='../data')
    
    # 2. Transform to tensors and normalize right away
    # This turns images into float tensors of shape (3, 28, 28) with values ~ [-1.0, 1.0]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # transforms.PILToTensor()
    ])
    
    print("Converting training data to tensors...")
    train_imgs = torch.stack([transform(img) for img, _ in train_dataset])   
    train_lbls = torch.tensor([label for _, label in train_dataset]) # Keeps original shape [N, 1]
    
    print("Converting validation data to tensors...")
    val_imgs = torch.stack([transform(img) for img, _ in val_dataset])
    val_lbls = torch.tensor([label for _, label in val_dataset])
    
    train_imgs = torch.cat([train_imgs, val_imgs], dim=0)
    train_lbls = torch.cat([train_lbls, val_lbls], dim=0)

    print("Converting test data to tensors...")
    test_imgs = torch.stack([transform(img) for img, _ in test_dataset])
    test_lbls = torch.tensor([label for _, label in test_dataset])

    # 3. Save into a single file dictionary
    payload = {
        'train_images': train_imgs,
        'train_labels': train_lbls,
        'test_images': test_imgs,
        'test_labels': test_lbls
    }
    
    torch.save(payload, '../data/'+output_name+'_'+str(size)+'data.pt')
    print("File successfully created and ready for distribution!")



if __name__ == "__main__":   
    generate_student_dataset()