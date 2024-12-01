import random
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt



class Network(object):
    #Sizes is number of neurons in layers of network
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    #Return the output of network for an input
    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    #Stochastic gradient descent to train neural network
    #training_data is an (x,y) tuple, x being the input, y being the desired result
    #Prints out accuracy of each epoch if test_data is provided
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            time2 = time.time()
            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2-time1))

    #Update weights and biases to a using gradient descent for a mini batch
    def update_mini_batch(self, mini_batch, l_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(l_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(l_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    #Returns tuple representing gradient for the cost function, nabla_b and nabla_w are layer by layer lists of numpy arrays
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    #Evaluates how the model did on test data, returns how many the network got correct
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedForward(x)), y)
                        for (x, y) in test_data]
        #print("Evaluate test results: ", test_results)
        return sum(int(x == y) for (x, y) in test_results)
    
    #Takes input picture and returns prediction
    def production_evaluate(self, test_data):
        test_results = (np.argmax(self.feedForward(test_data)))                
        return test_results

    #Get vector of partial derivative for output activations
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def print_random_test_digit(self, test_data):
        # Select a random test sample
        random_index = random.randint(0, len(test_data) - 1)
        x, y = test_data[random_index]

        # Reshape the input image from (784, 1) to (28, 28) for visualization
        img = x.reshape(28, 28)

        # Display the image
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {y}")
        plt.axis('off')
        plt.show()

#Sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#Derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#Saves neural network to a file
def save_network(network, filename):
    with open(filename, 'wb') as f:
        pickle.dump((network.biases, network.weights), f)