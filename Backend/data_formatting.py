import random
import numpy as np
from PIL import Image, ImageOps

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

#Randomly rotates image between max_angle and -max_angle
def random_rotation(image, max_angle=15):
    image = Image.fromarray(image)
    angle = random.uniform(-max_angle, max_angle)
    image = image.rotate(angle)
    return np.array(image)
    
#Randomly shifts image between max_shift angle and -max_shift angle
def random_shift(image, max_shift=0.1):
    image = Image.fromarray(image)
    width, height = image.size
    max_dx = int(max_shift * width)
    max_dy = int(max_shift * height)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    # Define the affine transformation matrix for shifting
    transformation_matrix = (1, 0, dx, 0, 1, dy)
    shifted_image = image.transform(
        image.size,
        Image.AFFINE,
        transformation_matrix,
        resample=Image.Resampling.NEAREST
    )
    return np.array(shifted_image)
    
#Randomly zooms image between max_zoom and min_zoom
def random_zoom(image, min_zoom=0.9, max_zoom=1.1):
    image = Image.fromarray(image)
    width, height = image.size
    zoom_factor = random.uniform(min_zoom, max_zoom)
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    cropped = resized.crop((0, 0, width, height))
    return np.array(cropped)
    
#Adds noise to image
def add_noise(image, noise_factor=0.1):
    np_image = np.array(image)
    noise = np.random.normal(0, noise_factor * 255, np_image.shape).astype(np.int32)
    noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)
