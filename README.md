# NeuralNetwork
Creating a neural network from scratch with numpy (no tensorflow)

I created this project to get a deeper understanding of how neural networks trully work

By watching a lot of youtube videos on the math behind the process I decided to create one myself.

1. This code takes in a 28x28 pixel image and converts it into a 1D numpy array.
2. The image vector is multiplied by a weights matrix and the resulting vector is summed with a bias vector. The new resulting vector has the ReLU activation function applied to it.
3. This process is repeated for the second hidden layer.
4. The process is once again repeated for the final output layer except instead of ReLU, softmax is applied. The final layer is a probility vector with 10 values.
5. The network takes the index position of the largest value and outputs it.

The network uses the back propagation algortihm to teach itself.
It does so by defining a cost function (in this case: Mean Squared Error) and calculating gradiants with respect to each weight and bias. It then multiplies the gradient with a learning rate and subtracts it from the current weights and biases.

To train the network, the MNIST data set is used.
