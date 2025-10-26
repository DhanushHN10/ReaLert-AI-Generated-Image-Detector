import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

ImgSize = 224

transform_train = transforms.Compose([
    transforms.Resize((ImgSize, ImgSize)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((ImgSize, ImgSize)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_dataset_clean(dataset_name="HPAI-BSC/SuSy-Dataset"):
    train_ds = load_dataset(dataset_name, split="train")
    val_ds = load_dataset(dataset_name, split="val")

    if train_ds.features["label"].__class__.__name__ == "ClassLabel":
        train_ds = train_ds.map(lambda x: {"label": int(x["label"])})
        val_ds = val_ds.map(lambda x: {"label": int(x["label"])})
    
    return train_ds, val_ds

def collate_fn(batch, is_train=True):
    transform = transform_train if is_train else transform_val
    images = torch.stack([transform(item["image"]) for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"pixel_values": images, "labels": labels}


def collate_fn_train(batch):
    return collate_fn(batch, is_train=True)

def collate_fn_val(batch):
    return collate_fn(batch, is_train=False)

def get_dataLoaders(train_ds, val_ds, batch_size=32):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_train, num_workers=5, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_val, num_workers=5, pin_memory=True)
    return train_loader, val_loader

if __name__ == "__main__":
    train_ds, val_ds = load_dataset_clean()
    train_loader, val_loader = get_dataLoaders(train_ds, val_ds, batch_size=8)
    batch = next(iter(train_loader))
    images, labels = batch["pixel_values"], batch["labels"]
    print(images.shape, labels.shape, labels.tolist())
