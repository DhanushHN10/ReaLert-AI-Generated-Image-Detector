import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
import sys
import os 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

script_dir = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.path.join(script_dir, "Final-AI-Generated-Image-Detector-Model")
print(f"Loading model from: {MODEL_PATH}")



model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()


ImgSize = 224
transform = transforms.Compose([
    transforms.Resize((ImgSize, ImgSize)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


label_map = {
    0: "Real",
    1: "dalle-3-images",
    2: "diffusiondb",
    3: "realisticSDXL",
    4: "midjourney-tti",
    5: "midjourney-images"  
}

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        all_probs = torch.softmax(outputs.logits, dim=1)[0] 
        
        print("--- Full Probability Distribution ---")
        for index, label_name in label_map.items():
            probability = all_probs[index].item()
            print(f"  {label_name}: {probability * 100:.2f}%")
        print("-------------------------------------")

        pred = torch.argmax(all_probs).item()
        confidence = all_probs[pred].item()

    label = label_map.get(pred, f"Class {pred}")
    print(f"Final Prediction: {label} (Confidence: {confidence * 100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict(image_path)