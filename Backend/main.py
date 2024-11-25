import mnist_loader
import network
import pickle
from network import save_network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

nn = network.Network([784, 128, 64, 10])
nn.SGD(training_data, 30, 32, 3.0, test_data=test_data)

#save_network(nn, "trained_network.pkl")

def load_network(filename):
    with open(filename, 'rb') as f:
        biases, weights = pickle.load(f)
    nn = network.Network([784, 30, 30, 10])
    nn.biases = biases
    nn.weights = weights
    return nn

#nn = load_network('trained_network.pkl')
