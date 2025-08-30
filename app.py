from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load your pre-trained model
MODEL_PATH = "model2.h5"  # Update with the path to your model
model = tf.keras.models.load_model(MODEL_PATH)

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))  # Resize image to model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        try:
            # Preprocess the image
            input_image = preprocess_image(file_path)

            # Perform prediction
            predictions = model.predict(input_image)
            predicted_class = np.argmax(predictions[0])  # Get the class with highest probability
            confidence = float(np.max(predictions[0]))

            # Map class indices to labels
            class_labels = {0: "mercury", 1: "healthy"}  # Update with your labels
            result = {
                "class": class_labels.get(predicted_class, "Unknown"),
                "confidence": confidence
            }

            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

        finally:
            # Clean up saved file
            os.remove(file_path)

    return jsonify({"error": "File not allowed"}), 400

@app.route('/info', methods=['GET'])
def info():
    return jsonify({"message": "Skin disease detection API", "endpoints": ["/predict", "/info"]})

if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    # Run the Flask app
    app.run(host='0.0.0.0', port=5001, debug=True)
