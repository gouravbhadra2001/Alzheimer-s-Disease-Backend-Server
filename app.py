import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)
CORS(app)

class_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
uploaded_image = None

@app.route('/', methods=['HEAD'])
def root():
    return '', 200


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    global uploaded_image
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        uploaded_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        uploaded_image = cv2.resize(uploaded_image, (300, 300))
        print("File Uploaded Successfully")
        return jsonify({'message': 'File uploaded successfully'}), 200

    except Exception as e:
        print(f"Error reading the image: {e}")
        return jsonify({'error': 'Error reading the image'}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global uploaded_image
    try:
        if uploaded_image is None:
            return jsonify({'error': 'No uploaded image'}), 400

        model = load_model('./model/AlzDisConvModel_InceptionV3_Hyper.h5')

        img_array = np.expand_dims(uploaded_image, axis=0)
        preprocessed_img = tf.keras.applications.inception_v3.preprocess_input(img_array)

        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

        print(f"Predicted class index: {predicted_class_index}")
        return jsonify({'prediction': predicted_class_label, 'confidence': float(predictions[0][predicted_class_index])}), 200

    except Exception as e:
        print(f"Error predicting: {e}")
        return jsonify({'error': 'Error predicting'}), 500
    finally:
        uploaded_image = None

if __name__ == '__main__':
    app.run(debug=True)
