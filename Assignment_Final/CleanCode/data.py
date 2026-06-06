import torch
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(pt_path='../data/histology_data.pt', batch_size=128):
    data_dict = torch.load(pt_path)
    
    train_dataset = TensorDataset(data_dict['train_images'], data_dict['train_labels'])
    val_dataset = TensorDataset(data_dict['val_images'], data_dict['val_labels'])
    test_dataset = TensorDataset(data_dict['test_images'], data_dict['test_labels'])
    
    # 3. Create the data loaders for execution
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader