import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoModelForImageClassification
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


model = AutoModelForImageClassification.from_pretrained(
    "./Final-AI-Generated-Image-Detector-Model"
).to(device)
model.eval()


ImgSize = 224
transform_test = transforms.Compose([
    transforms.Resize((ImgSize, ImgSize)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


test_ds = load_dataset("HPAI-BSC/SuSy-Dataset", split="test")


if test_ds.features["label"].__class__.__name__ == "ClassLabel":
    test_ds = test_ds.map(lambda x: {"label": int(x["label"])})


def collate_fn(batch):
    images = torch.stack([transform_test(item["image"]) for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"pixel_values": images, "labels": labels}


test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)


correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(images)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
