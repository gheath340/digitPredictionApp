import numpy as np

#Changes label from single number to one hot encoded list
def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((num_classes, 1))
    one_hot[y] = 1
    return one_hot

#Normalizes training data with one_hot_encoded labels
def normalize_training_data(data):
    return [(np.reshape(x, (784, 1)) / 255.0, one_hot_encode(y)) for x, y in data]

#Normalizes test data
def normalize_test_data(data):
    return [(np.reshape(x, (784, 1)) / 255.0, y) for x, y in data]