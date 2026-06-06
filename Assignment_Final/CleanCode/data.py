import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

def get_pathmnist_loaders(batch_size=64):
    data_flag = 'pathmnist'
    data_flag = 'tissuemnist'
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
    
    train_dataset = DataClass(split='train', transform=transform_train, download=True, root='../data')
    val_dataset = DataClass(split='val', transform=transform_val, download=True, root='../data')
    test_dataset = DataClass(split='test', transform=transform_val, download=True, root='../data')
    
    # Create the standard PyTorch Dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader