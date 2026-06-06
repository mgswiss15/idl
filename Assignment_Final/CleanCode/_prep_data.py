import torch
import medmnist
from torchvision import transforms

def generate_student_dataset():
    # 1. Download/Load raw data
    train_dataset = medmnist.PathMNIST(split='train', download=True)
    val_dataset = medmnist.PathMNIST(split='val', download=True)
    
    # 2. Transform to tensors and normalize right away
    # This turns images into float tensors of shape (3, 28, 28) with values ~ [-1.0, 1.0]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print("Converting training data to tensors...")
    train_imgs = torch.stack([transform(img) for img, _ in train_dataset])
    train_lbls = torch.tensor([label for _, label in train_dataset]) # Keeps original shape [N, 1]
    
    print("Converting validation data to tensors...")
    val_imgs = torch.stack([transform(img) for img, _ in val_dataset])
    val_lbls = torch.tensor([label for _, label in val_dataset])
    
    # 3. Save into a single file dictionary
    payload = {
        'train_images': train_imgs,
        'train_labels': train_lbls,
        'val_images': val_imgs,
        'val_labels': val_lbls
    }
    
    torch.save(payload, 'tissue_data.pt')
    print("🎉 File 'tissue_data.pt' successfully created and ready for distribution!")

if __name__ == "__main__":
    generate_student_dataset()