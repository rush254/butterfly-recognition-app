from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import base64
import numpy as np
from io import BytesIO
from werkzeug.utils import secure_filename
from src.model_processing import *
from src.azure_blob import *


app = Flask(__name__)

# Set a max image size
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Set up Azure Blob Storage connection
connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = "butterfly-recognition"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # get filename
        filename = secure_filename(file.filename)
        
        predicted_label = predict_class(load_and_preprocess_image(image_path))

        # Store the image in the 'images' folder in blob storage
        img_url = upload_image_to_blob(file.read(), filename, connect_str, container_name)

        return jsonify({
            'predicted_label': predicted_label,
            'image_url': img_url
        })

    else:
        return jsonify({'error': 'File type not allowed'}), 400



if __name__ == '__main__':
    app.run(debug=True)
