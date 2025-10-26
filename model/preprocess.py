import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

def convert_labels_binary(label):
    if label == 0:
        return 0
    else:
        return 1

class SuSYDataset(Dataset):
    
    def __init__(self, hf_dataset, transform=None,
                #  binary=True
                 ):
        self.dataset = hf_dataset
        self.transform = transform
        # self.binary = binary
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # if self.binary:
        #     label = 0 if label == 0 else 1
            
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataloaders(batch_size=8):    
    dataset = load_dataset("HPAI-BSC/SuSY-Dataset")
            
    train_dataset = SuSYDataset(dataset['train'], transform=get_train_transforms())
    val_dataset = SuSYDataset(dataset['val'], transform=get_val_transforms())
    test_dataset = SuSYDataset(dataset['test'], transform=get_val_transforms())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader
