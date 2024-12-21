from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Load the trained model once when the server starts
MODEL_PATH = './final_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    # Resize the image to 252x252
    image = image.resize((252, 252))
    # Convert to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    # Add batch and channel dimensions
    img_array = np.expand_dims(img_array, axis=[0, -1])
    return img_array

def postprocess_mask(mask):
    # Convert probabilities to binary mask
    return (mask > 0.5).astype(np.uint8) * 255

def image_to_base64(img_array):
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    try:
        # Open the image
        image = Image.open(file.stream)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(preprocessed_image)
        
        # Postprocess the prediction
        liver_mask = postprocess_mask(prediction[0, :, :, 0])
        tumor_mask = postprocess_mask(prediction[0, :, :, 1])
        
        # Convert masks to base64
        liver_mask_b64 = image_to_base64(liver_mask)
        tumor_mask_b64 = image_to_base64(tumor_mask)
        
        return jsonify({
            'liver_mask': liver_mask_b64,
            'tumor_mask': tumor_mask_b64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure the static folder exists
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host='0.0.0.0', port=5000, debug=True)
