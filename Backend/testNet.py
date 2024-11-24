import network
import pickle
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def load_network(filename):
    #Open the trained network and retrieve weights and biases
    #Initialize network with neurons, initialize weights and biases
    with open(filename, 'rb') as f:
        biases, weights = pickle.load(f)
    nn = network.Network([784, 30, 30, 10])
    nn.biases = biases
    nn.weights = weights
    return nn

def preprocess_image(image_path):
    #Convert to grayscale, resize image, reshape array, normalize to [0,1]
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(784, 1)
    img_array = img_array / 255.0

    plt.imshow(img, cmap='gray')
    plt.title("Preprocessed Image")
    plt.axis('off')
    plt.show()
    return img_array

nn = load_network('trained_network.pkl')

input_data = preprocess_image("eight.png")
prediction = nn.prodEvaluate(input_data)
print(f"Prediction: {prediction}")
