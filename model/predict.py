import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os

model_path = "efficientnet_b4_epoch16_newnew.pt"
num_classes = 6
image_size = 380

CLASS_NAMES = {
    0: "Real",
    1: "dalle-3-images",
    2: "diffusiondb", 
    3: "realisticSDXL",
    4: "midjourney-tti",
    5: "midjourney-images"
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(model_path, num_classes, device):

    print("📦 Loading model...")
    
    model = timm.create_model('tf_efficientnet_b4', pretrained=False, num_classes=num_classes)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"✅ Model loaded from {model_path}\n")
    
    return model


def get_transforms():

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_image(image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    transform = get_transforms()
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, 3, 380, 380]
    
    return image_tensor

def predict_image(model, image_path, device):
    # Preprocess
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)  # Convert to probabilities
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    confidence_percent = confidence.item() * 100
    all_probs = probabilities[0].cpu().numpy()
    
    return predicted_class, confidence_percent, all_probs


def test_single_image(model, image_path, device):

    print(f"🖼️  Testing: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}\n")
        return
    
    # Predict
    try:
        predicted_class, confidence, probabilities = predict_image(model, image_path, device)
        
        # Print results
        print(f"✅ Prediction: {CLASS_NAMES[predicted_class]}")
        print(f"🎯 Confidence: {confidence:.2f}%")
        print(f"\n📊 All Class Probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"   {CLASS_NAMES[i]}: {prob*100:.2f}%")
        print()
        
    except Exception as e:
        print(f"❌ Error processing image: {e}\n")

def test_multiple_images(model, image_paths, device):
    """
    Test model on multiple images
    """
    print("="*70)
    print("🧪 TESTING MODEL ON YOUR IMAGES")
    print("="*70 + "\n")
    
    for image_path in image_paths:
        test_single_image(model, image_path, device)
        print("-"*70 + "\n")


def test_folder(model, folder_path, device):    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    image_files = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        print("❌ No images found in folder\n")
        return
    
    print(f"Found {len(image_files)} images\n")
    print("="*70 + "\n")
    
    # Test each image
    test_multiple_images(model, image_files, device)



if __name__ == '__main__':
    
    # Load model
    model = load_model(model_path, num_classes, device)
    
    # ============================================
    # OPTION 1: Test on specific images
    # ============================================
    # Uncomment and add your image paths
    
    
    
    # test_multiple_images(model, image_paths, device)
    
    
    # ============================================
    # OPTION 2: Test on entire folder
    # ============================================
    # Uncomment to test all images in a folder
    
    test_folder(model, "img/", device)
    
    
    # ============================================
    # OPTION 3: Interactive testing
    # ============================================
    # Uncomment for interactive mode
    
    # while True:
    #     image_path = input("\n🖼️  Enter image path (or 'quit' to exit): ").strip()
    #     if image_path.lower() == 'quit':
    #         break
    #     test_single_image(model, image_path, device)
    
