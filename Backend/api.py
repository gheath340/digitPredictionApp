from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import network
import pickle
from PIL import Image, ImageOps
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def load_network(filename):
    #Open the trained network and retrieve weights and biases
    #Initialize network with neurons, initialize weights and biases
    with open(filename, 'rb') as f:
        biases, weights = pickle.load(f)
    nn = network.Network([784, 30, 30, 10])
    nn.biases = biases
    nn.weights = weights
    return nn

def brighten_image(pixel):
    #brighten everything that isnt black
    if pixel == 0:
        return pixel
    else:
        return 255

def preprocess_image(image):
    #Convert to grayscale and invert
    img = Image.open(BytesIO(image))
    img = img.convert('L')
    img = ImageOps.invert(img)
    #resize
    img = img.resize((28, 28))
    #turn into array and brighten everything that isnt black
    img_array = np.array(img)
    vectorized_brighten = np.vectorize(brighten_image)
    img_array = vectorized_brighten(img_array)
    brighter_image = Image.fromarray(np.uint8(img_array))
    #reshape and normalize pixel values between 0 and 1
    img_array = img_array.reshape(784, 1)
    img_array = img_array / 255.0

    return img_array

nn = load_network('trained_network.pkl')


@app.route('/predict', methods=['OPTIONS', 'POST'])
def predict_route():
    #handle preflight request
    if request.method == 'OPTIONS':
        response = jsonify({"message": "CORS preflight successful"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 204
    #get base64 string, remove start, decode base64, preprocess and predict
    data = request.get_json()
    base64_string = data["image"]
    if data["image"].startswith("data:image/png;base64,"):
        base64_string = base64_string.replace("data:image/png;base64,", "")
    image = base64.b64decode(base64_string)

    image_data = preprocess_image(image)
    prediction = int(nn.production_evaluate(image_data))

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
