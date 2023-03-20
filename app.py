import io
import base64
import PIL
import pandas as pd
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
import pickle

import skimage

# Initialize Flask application
app = Flask(__name__)

# Load SVM model
with open("model/model.pkl", 'rb') as file:  
    clf = pickle.load(file)

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    img_data = request.form['img_data']

    # Decode image from base64
    img = base64.b64decode(img_data.split(',')[1])

    # Convert image to numpy array
    img = skimage.io.imread(io.BytesIO(img))

    # Convert image to grayscale and resize to 28x28
    img = PIL.Image.fromarray(img).convert('L').resize((28, 28))

    # Flatten image to 1D array
    img = np.array(img).flatten()

    # Make prediction using SVM model
    pred = clf.predict([img])[0]

    # Return prediction result as JSON
    return {'result': int(pred)}

if __name__ == '__main__':
    app.run(debug=True)
