import numpy as np
from random import uniform


class NeuralNetwork:
    def __init__(self, layer_one_node_count, layer_two_node_count):
        self.input_vector = np.empty(784)

        # Functions to create list and 2D array with random values
        # I got better results using this instead of np.random.rand
        random_list = lambda length: [uniform(-1.0, 1.0) for _ in range(length)]
        random_array = lambda row, column: np.array([random_list(column) for _ in range(0, row)])

        # Create weight matrices with random values
        self.layer_one_weight_matrix = random_array(layer_one_node_count, 784)
        self.layer_two_weight_matrix = random_array(layer_two_node_count, layer_one_node_count)
        self.layer_three_weight_matrix = random_array(10, layer_two_node_count)

        # Create bias vectors with random values
        self.layer_one_bias_vector = np.array(random_list(layer_one_node_count))
        self.layer_two_bias_vector = np.array(random_list(layer_two_node_count))
        self.layer_three_bias_vector = np.array(random_list(10))

        # Initialize vectors for each node in a layer
        self.layer_one_node_vector = np.empty(layer_one_node_count, dtype=float)
        self.layer_two_node_vector = np.empty(layer_two_node_count, dtype=float)
        self.output_node_vector = np.empty(10, dtype=float)

    # Function that performs forward propagation
    # Uses ReLU (np.maximum) to activate layers 1 and 2
    # USes softmax to activate layer 3
    def propagate(self, input_vector):
        softmax = lambda vector: np.exp(vector - np.max(vector)) / np.exp(vector - np.max(vector)).sum()

        self.input_vector = input_vector
        self.layer_one_node_vector = np.maximum(0, np.add(self.layer_one_weight_matrix.dot(input_vector), self.layer_one_bias_vector))
        self.layer_two_node_vector = np.maximum(0, np.add(self.layer_two_weight_matrix.dot(self.layer_one_node_vector),self.layer_two_bias_vector))
        self.output_node_vector = softmax(np.add(self.layer_three_weight_matrix.dot(self.layer_two_node_vector), self.layer_three_bias_vector))

    # Function that performs back propagation
    def learn(self, desired_vector):
        ReLU_derivative = lambda vector: np.where(vector > 0, 1, 0)

        learning_rate = 0.0001

        # Calculate derivative of error term for each layer
        # Also happens to be gradient for bias vectors
        layer_three_error_vector = 2 * np.subtract(self.output_node_vector, desired_vector)
        layer_two_error_vector = np.multiply(np.dot(layer_three_error_vector, self.layer_three_weight_matrix),ReLU_derivative(np.add(self.layer_two_weight_matrix.dot(self.layer_one_node_vector),self.layer_two_bias_vector)))
        layer_one_error_vector = np.multiply(np.dot(layer_two_error_vector, self.layer_two_weight_matrix), ReLU_derivative(np.add(self.layer_one_weight_matrix.dot(self.input_vector), self.layer_one_bias_vector)))

        # Calculate gradiant for weight matrices
        layer_three_weight_gradiant = np.outer(layer_three_error_vector, self.layer_two_node_vector.T)
        layer_two_weight_gradiant = np.outer(layer_two_error_vector, self.layer_one_node_vector.T)
        layer_one_weight_gradiant = np.outer(layer_one_error_vector, self.input_vector.T)

        # Update weight matrices with gradients
        self.layer_three_weight_matrix -= learning_rate * layer_three_weight_gradiant
        self.layer_two_weight_matrix -= learning_rate * layer_two_weight_gradiant
        self.layer_one_weight_matrix -= learning_rate * layer_one_weight_gradiant

        # Update bias vectors with gradients
        self.layer_three_bias_vector -= learning_rate * layer_three_error_vector
        self.layer_two_bias_vector -= learning_rate * layer_two_error_vector
        self.layer_one_bias_vector -= learning_rate * layer_one_error_vector
