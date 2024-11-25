import random
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import random
import time
import pickle
import matplotlib.pyplot as plt

# Adam Optimizer class
# class AdamOptimizer:
#     def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.epsilon = epsilon
#         self.m = None  # First moment
#         self.v = None  # Second moment
#         self.t = 0  # Time step

#     def update(self, params, grads):
#         if self.m is None:
#             self.m = [np.zeros_like(param) for param in params]  # Initialize m
#         if self.v is None:
#             self.v = [np.zeros_like(param) for param in params]  # Initialize v

#         self.t += 1  # Increment the timestep
#         updated_params = []

#         for i in range(len(params)):
#             grad = grads[i]
#             m_t = self.beta1 * self.m[i] + (1 - self.beta1) * grad  # First moment estimate
#             v_t = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)  # Second moment estimate

#             # Bias correction
#             m_hat = m_t / (1 - self.beta1 ** self.t)
#             v_hat = v_t / (1 - self.beta2 ** self.t)

#             # Update parameters
#             param_update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
#             updated_param = params[i] - param_update
#             updated_params.append(updated_param)

#             # Update m and v for next iteration
#             self.m[i] = m_t
#             self.v[i] = v_t

#         return updated_params


# class Network(object):
#     def __init__(self, sizes):
#         self.num_layers = len(sizes)
#         self.sizes = sizes
#         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#         self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
#         # Initialize Adam optimizer
#         self.adam_optimizer = AdamOptimizer()

#     def feedForward(self, a):
#         """Feed the input through the network and return the output."""
#         for b, w in zip(self.biases, self.weights):
#             print(f"Before feedforward: a.shape = {a.shape}, w.shape = {w.shape}, b.shape = {b.shape}")
#             a = sigmoid(np.dot(w, a) + b)  # Ensure a is (n, 1)
#             print(f"After feedforward: a.shape = {a.shape}")
#         return a

#     def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
#         if test_data:
#             n_test = len(test_data)
#         n = len(training_data)
#         for j in range(epochs):
#             time1 = time.time()
#             random.shuffle(training_data)
#             mini_batches = [
#                 training_data[k:k + mini_batch_size]
#                 for k in range(0, n, mini_batch_size)
#             ]
#             for mini_batch in mini_batches:
#                 self.update_mini_batch(mini_batch, eta)
#             time2 = time.time()
#             if test_data:
#                 print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}, took {time2 - time1:.2f} seconds")
#             else:
#                 print(f"Epoch {j} complete in {time2 - time1:.2f} seconds")
#             self.print_random_test_digit(test_data)

#     def update_mini_batch(self, mini_batch, eta):
#         """Update the network's weights and biases by applying Adam optimizer to a mini-batch."""
#         nabla_b = [np.zeros(b.shape) for b in self.biases]
#         nabla_w = [np.zeros(w.shape) for w in self.weights]
#         for x, y in mini_batch:
#             # Ensure x is reshaped as (784, 1)
#             x = x.reshape(-1, 1)
#             delta_nabla_b, delta_nabla_w = self.backprop(x, y)
#             nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
#             nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
#         # Update weights and biases using Adam optimizer
#         self.weights = self.adam_optimizer.update(self.weights, nabla_w)
#         self.biases = self.adam_optimizer.update(self.biases, nabla_b)

#     def backprop(self, x, y):
#         nabla_b = [np.zeros(b.shape) for b in self.biases]
#         nabla_w = [np.zeros(w.shape) for w in self.weights]
        
#         # Reshape input x to ensure it is in the correct shape (784, 1) for MNIST
#         activation = x  # (784, 1) shape
#         activations = [activation]  # list to store all the activations, layer by layer
#         zs = []  # list to store all the z vectors, layer by layer
#         print(f"Starting backprop: activation.shape = {activation.shape}")
#         for b, w in zip(self.biases, self.weights):
#             z = np.dot(w, activation) + b
#             zs.append(z)
#             activation = sigmoid(z)
#             activations.append(activation)
#             print(f"After layer: activation.shape = {activation.shape}, z.shape = {z.shape}")
        
#         # backward pass
#         delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
#         nabla_b[-1] = delta
#         nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
#         # Backpropagation for the other layers
#         for l in range(2, self.num_layers):
#             z = zs[-l]
#             sp = sigmoid_prime(z)
#             delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
#             nabla_b[-l] = delta
#             nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
#         return nabla_b, nabla_w

#     def evaluate(self, test_data):
#         test_results = [(np.argmax(self.feedForward(x)), y) for x, y in test_data]
#         return sum(int(x == y) for x, y in test_results)

#     def cost_derivative(self, output_activations, y):
#         return output_activations - y
    
#     def print_random_test_digit(self, test_data):
#         random_index = random.randint(0, len(test_data) - 1)
#         x, y = test_data[random_index]
#         img = x.reshape(28, 28)
#         plt.imshow(img, cmap='gray')
#         plt.title(f"Label: {y}")
#         plt.axis('off')
#         plt.show()

# def sigmoid(z):
#     return 1.0 / (1.0 + np.exp(-z))

# def sigmoid_prime(z):
#     return sigmoid(z) * (1 - sigmoid(z))

# def save_network(network, filename):
#     with open(filename, 'wb') as f:
#         pickle.dump((network.biases, network.weights), f)





class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedForward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
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
            #self.print_random_test_digit(test_data)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
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
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedForward(x)), y)
                        for (x, y) in test_data]
        #print("Evaluate test results: ", test_results)
        return sum(int(x == y) for (x, y) in test_results)
    
    def prodEvaluate(self, test_data):
        test_results = (np.argmax(self.feedForward(test_data)))
                        
        return test_results

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives for the output activations."""
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

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def save_network(network, filename):
    with open(filename, 'wb') as f:
        pickle.dump((network.biases, network.weights), f)