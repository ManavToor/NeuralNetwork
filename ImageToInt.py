import cv2

import numpy as np
from keras.datasets import mnist

from NeuralNetwork import NeuralNetwork

vector_to_int = lambda input_vector: np.argmax(input_vector)
int_to_vector = lambda input_int: np.array([1 if i == input_int else 0 for i in range(10)])

# Load training and testing data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Initialize network
CharacterRecognizer = NeuralNetwork(128, 128)

# Train network
for i in range(0, len(train_x)):
    CharacterRecognizer.propagate(train_x[i].flatten() / 255)
    CharacterRecognizer.learn(int_to_vector(train_y[i]))

# Turn image into a vector, only focus on B values in RGB (does not actually mater which one we chose)
image_array = cv2.imread('img.png')[:,:,0]
image_vector = image_array.flatten() / 255

# Print accuracy of network
print(f'Number is {vector_to_int(CharacterRecognizer.propagate(image_vector))}')
