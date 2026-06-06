import torch
from torchvision import transforms
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size):
    d_path = Path(data_path) / f"{data}_data.pt"
    data_dict = torch.load(d_path)

    # normalization
    mean, std = 0.5, 0.5

    train_dataset = TensorDataset((data_dict['train_images']-mean)/std, data_dict['train_labels'])
    val_dataset = TensorDataset((data_dict['val_images']-mean)/std, data_dict['val_labels'])
    test_dataset = TensorDataset((data_dict['test_images']-mean)/std, data_dict['test_labels'])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader