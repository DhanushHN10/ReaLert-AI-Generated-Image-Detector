import atexit
import shutil
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import requests
from io import BytesIO
import traceback
import os
import uuid

app = Flask(__name__)
CORS(app)

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

TEMP_DIR = "temp_images"

def setup_temp_dir():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

def cleanup_temp_dir():
    """Delete temp directory and all contents"""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

setup_temp_dir()
atexit.register(cleanup_temp_dir)

def get_simple_category(class_id):
    if class_id == 0:
        return "Real"
    else:
        return "AI-Generated"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = timm.create_model('tf_efficientnet_b4', pretrained=False, num_classes=num_classes)

checkpoint = torch.load(model_path, map_location=device)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

def get_transform():
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
def download_and_save_image(url):
    try:
        unique_id = str(uuid.uuid4())
        temp_filepath = os.path.join(TEMP_DIR, f"{unique_id}.jpg")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image.save(temp_filepath, 'JPEG')        
        return image, temp_filepath
        
    except Exception as e:
        raise Exception(f"Failed to download image: {str(e)}")
    
def delete_temp_image(filepath):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Failed to delete {filepath}: {str(e)}")
        
    
def preprocess_image(image):
    transform = get_transform()
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def predict_image(image):
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    confidence_percent = confidence.item() * 100
    all_probs = probabilities[0].cpu().numpy()
    
    result = {
        'class_id': predicted_class,
        'class_name': CLASS_NAMES[predicted_class],
        'category': get_simple_category(predicted_class),
        'confidence': round(confidence_percent, 2),
        'probabilities': {
            CLASS_NAMES[i]: round(float(all_probs[i]) * 100, 2) 
            for i in range(num_classes)
        }
    }
    
    return result

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'AI Image Detector API is running',
        'model': 'EfficientNet-B4',
        'classes': num_classes
    })

@app.route('/predict', methods=['POST'])
def predict():
    temp_filepath = None
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing image_url in request body'
            }), 400
        
        image_url = data['image']
        
        print(f"Received request for: {image_url}")
        
        image, temp_filepath = download_and_save_image(image_url)
        
        # Predict
        prediction = predict_image(image)
        print(f"Prediction: {prediction['category']} ({prediction['confidence']:.2f}%)")
        
        return jsonify({
            'success': True,
            'image_url': image_url,
            'prediction': prediction
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# @app.route('/predict-batch', methods=['POST'])
# def predict_batch():
#     try:
#         data = request.get_json()
        
#         if not data or 'image_urls' not in data:
#             return jsonify({
#                 'success': False,
#                 'error': 'Missing image_urls in request body'
#             }), 400
        
#         image_urls = data['image_urls']
        
#         if not isinstance(image_urls, list):
#             return jsonify({
#                 'success': False,
#                 'error': 'image_urls must be a list'
#             }), 400
        
#         # Limit batch size
#         if len(image_urls) > 10:
#             return jsonify({
#                 'success': False,
#                 'error': 'Maximum 10 images per batch'
#             }), 400
        
#         print(f"Received batch request for {len(image_urls)} images")
        
#         results = []
        
#         for url in image_urls:
#             try:
#                 image = download_image(url)
#                 prediction = predict_image(image)
                
#                 results.append({
#                     'success': True,
#                     'image_url': url,
#                     'prediction': prediction
#                 })
#             except Exception as e:
#                 results.append({
#                     'success': False,
#                     'image_url': url,
#                     'error': str(e)
#                 })
        
#         return jsonify({
#             'success': True,
#             'results': results
#         })
    
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         traceback.print_exc()
        
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

if __name__ == '__main__':
    print(f"Model: EfficientNet-B4")
    print(f"Classes: {num_classes}")
    print(f"Device: {device}")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',  # Accessible from other devices
        port=6000,        # Port number
        debug=False       # Set to False in production
    )