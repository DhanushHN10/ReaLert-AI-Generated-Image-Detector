import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
import os
import io 
import requests 
from flask import Flask, request, jsonify


app = Flask(__name__)


print("Loading model and setting up device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

script_dir = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(script_dir, "Final-AI-Generated-Image-Detector-Model")
print(f"Loading model from: {MODEL_PATH}")

try:
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval() 
except Exception as e:
    print(f"--- FATAL ERROR: Could not load model. ---")
    print(e)
 

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
print("Model loaded successfully. Ready to accept requests.")



@app.route("/", methods=["GET"])
def home_route():
   
    return "API is up and running. Use the /predict endpoint to make predictions."



@app.route("/predict", methods=["POST"])
def predict_route():

    data = request.get_json()
   
    if not data or 'image' not in data: 
        return jsonify({"error": "Missing 'image' url in JSON payload"}), 400


    image_url = data['image'] 
    print(image_url)

    try:
        
        response = requests.get(image_url)
        
        response.raise_for_status() 
        
    
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

    except requests.exceptions.RequestException as e:
        print("hello")
        return jsonify({"error": f"Failed to download or open image: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred processing the image: {str(e)}"}), 500


    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        all_probs = torch.softmax(outputs.logits, dim=1)[0]
        
        probabilities = {}
        for index, label_name in label_map.items():
            probability = all_probs[index].item()
            probabilities[label_name] = round(probability * 100, 2)

        pred_index = torch.argmax(all_probs).item()
        confidence = round(all_probs[pred_index].item() * 100, 2) 
        pred_label = label_map.get(pred_index, f"Class {pred_index}")

   
        result = {
            "image_url": image_url, 
            "prediction": {
                "category": pred_label,
                "class_id": pred_index,
                "class_name": pred_label,
                "confidence": confidence,
                "probabilities": probabilities
            },
            "success": True
        }
    
    return jsonify(result)


if __name__ == "__main__":
 
    app.run(debug=True, host='0.0.0.0', port=5000)