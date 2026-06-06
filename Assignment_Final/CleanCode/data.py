import torch
from torchvision import transforms
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):
    d_path = Path(data_path) / f"{data}.pt"
    data_dict = torch.load(d_path)

    # normalization
    mean, std = 0.5, 0.5
    # mean, std = 0, 1

    total_samples = data_dict['train_images'].shape[0]
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    train_data = data_dict['train_images'][:train_size]
    train_labels = data_dict['train_labels'][:train_size]
    val_data = data_dict['train_images'][train_size:]
    val_labels = data_dict['train_labels'][train_size:]

    train_dataset = TensorDataset((train_data-mean)/std, train_labels)
    val_dataset = TensorDataset((val_data-mean)/std, val_labels)
    test_dataset = TensorDataset((data_dict['test_images']-mean)/std, data_dict['test_labels'])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader