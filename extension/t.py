from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Dummy function to simulate an AI prediction
def predict_image(image_url):
    # This would normally call your ML model
    return {
        "category": "AI-generated",
        "confidence": 87.42
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_url = data.get("url")

        if not image_url:
            return jsonify({
                "success": False,
                "error": "Missing 'image_url' in request body"
            }), 400

        prediction = predict_image(image_url)
        print(f"Prediction: {prediction['category']} ({prediction['confidence']:.2f}%)")

        return jsonify({
            "success": True,
            "image_url": image_url,
            "prediction": prediction
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)
    print("hi")

