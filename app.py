from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

import base64
import json
from io import BytesIO

# Keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img, model):
    return model.predict(np.array([img]))


@app.route('/', methods=['GET'])
def index():
    # Main page
    return "Hello World"


@app.route('/predict/', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        img = image.img_to_array(image.load_img(
            BytesIO(base64.b64decode(request.form['b64'])), target_size=(75, 100, 3)))

        # Make prediction
        preds = model_predict(img, model)
        print(preds)
        a = ['nv', 'mel', 'bkl', 'bcc', 'akie',
             'vas', 'df']           # Convert to string
        return jsonify({"prediction": a[int(np.argmax(preds[0]))]})
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
