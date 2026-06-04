import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

def get_pathmnist_loaders(batch_size=64):
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Instantiate the MedMNIST datasets
    DataClass = getattr(medmnist, info['python_class'])
    
    train_dataset = DataClass(split='train', transform=transform_train, download=True, path='../../data')
    val_dataset = DataClass(split='val', transform=transform_val, download=True, path='../../data')
    test_dataset = DataClass(split='test', transform=transform_val, download=True, path='../../data')
    
    # Create the standard PyTorch Dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Quick shape verification check
    t_loader, v_loader, test_loader = get_pathmnist_loaders(batch_size=4)
    images, labels = next(iter(t_loader))
    print(f"Dataset Verification Successful!")
    print(f"Images Tensor Shape: {images.shape}")  # Expect: [4, 3, 28, 28]
    print(f"Labels Tensor Shape: {labels.shape}")  # Expect: [4, 1]