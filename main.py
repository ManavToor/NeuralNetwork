import numpy as np
from keras.datasets import mnist

from NeuralNetwork import NeuralNetwork

vector_to_int = lambda input_vector: np.argmax(input_vector)
int_to_vector = lambda input_int: np.array([1 if i == input_int else 0 for i in range(10)])

# Load training and testing data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Initialize network
CharacterRecognizer = NeuralNetwork(128, 64)

# Train network
for i in range(0, len(train_x)):
    CharacterRecognizer.propagate(train_x[i].flatten() / 255)
    CharacterRecognizer.learn(int_to_vector(train_y[i]))

# variable for keeping track of correct outputs by network
dubs = 0

# Test network
for i in range(0, len(test_x)):
    CharacterRecognizer.propagate(test_x[i].flatten() / 255)

    if vector_to_int(CharacterRecognizer.output_node_vector) == test_y[i]:
        dubs += 1

# Print accuracy of network
print(f'Network is {dubs/len(test_x) * 100}% accurate')
