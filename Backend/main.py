import network
import pickle
import random
import numpy as np
import data_formatting as df
from network import save_network


#Loads mnist dataset
data = np.load('mnist.npz')

#Divides training and test data into tuples where x is image data and y is label
training_data = list(zip(data['x_train'], data['y_train']))
test_data = list(zip(data['x_test'], data['y_test']))

function_list = [df.random_rotation, df.random_shift, df.random_zoom, df.add_noise, None]
for image in training_data:
    random_function = random.choice(function_list)
    if random_function:
        random_function(image[0])

#Normalize data
training_data = df.normalize_training_data(training_data)
test_data = df.normalize_test_data(test_data)

#Creates and trains network
nn = network.Network([784, 32, 16, 10])
nn.SGD(training_data, 30, 32, 3.0, test_data=test_data)

save_network(nn, "trained_network.pkl")

#Loads saved network
def load_network(filename):
    with open(filename, 'rb') as f:
        biases, weights = pickle.load(f)
    nn = network.Network([784, 30, 30, 10])
    nn.biases = biases
    nn.weights = weights
    return nn

#nn = load_network('trained_network.pkl')
