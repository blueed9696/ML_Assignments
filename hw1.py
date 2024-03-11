class Perceptron:
    def __init__(self, num_inputs, init_weight, init_bias, activation_fcn='tanh'):
        # Init Weights (All to 0.1)
        self.weights = [init_weight for _ in range(num_inputs)]
        # Init Bias
        self.bias = init_bias
        # Init activation fcn
        self.activation_fcn = activation_fcn

    def predict(self, inputs):
        # Weighted Sum
        weighted_sum = sum(inputs[i] * self.weights[i] for i in range(len(inputs))) + self.bias

        # Apply activation function
        if self.activation_fcn == 'sigmoid':
            return self.sigmoid(weighted_sum)
        elif self.activation_fcn == 'tanh':
            return self.tanh(weighted_sum)
        elif self.activation_fcn == 'relu':
            return self.relu(weighted_sum)
        else:
            raise ValueError("Wrong Activation Function Input!!")

    # -------------------- Activation Functions ------------------
    # Exponential expanded with taylor series
    def exponential(self, x):
        return 1 + x + (x ** 2) / 2 + (x ** 3) / 6 + (x ** 4) / 24 + (x ** 5) / 120

    # Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + self.exponential(-x))

    # Tanh
    def tanh(self, x):
        return (self.exponential(x) - self.exponential(-x)) / (self.exponential(x) + self.exponential(-x))

    def relu(self, x):
        return max(0, x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Create input layer perceptrons
        self.input_layer = [Perceptron(num_inputs=1, init_weight=0.1, init_bias=0) for _ in range(input_size)]

        # Create hidden layer perceptrons
        self.hidden_layer = [Perceptron(num_inputs=input_size, init_weight=0.1, init_bias=0) for _ in range(hidden_size)]

        # Create output layer perceptrons
        self.output_layer = [Perceptron(num_inputs=hidden_size, init_weight=0.1, init_bias=0) for _ in range(output_size)]

    def forward(self, inputs):
        # Forward pass through the neural network
        hidden_layer_outputs = [perceptron.predict(inputs) for perceptron in self.hidden_layer]
        output_layer_outputs = [perceptron.predict(hidden_layer_outputs) for perceptron in self.output_layer]
        return output_layer_outputs

    def train(self, inputs, labels, learning_rate=0.1, epochs=1):
        for _ in range(epochs):
            for input_data, label in zip(inputs, labels):
                # Forward pass
                hidden_layer_outputs = [perceptron.predict(input_data) for perceptron in self.hidden_layer]
                output_layer_outputs = [perceptron.predict(hidden_layer_outputs) for perceptron in self.output_layer]

                # Calculate output layer errors
                output_errors = [label[i] - output_layer_outputs[i] for i in range(self.output_size)]

                # Backpropagation - update weights and biases of output layer
                for i in range(self.output_size):
                    perceptron = self.output_layer[i]
                    perceptron.weights = [perceptron.weights[j] + learning_rate * output_errors[i] * hidden_layer_outputs[j]
                                          for j in range(len(perceptron.weights))]
                    perceptron.bias += learning_rate * output_errors[i]

                # Calculate hidden layer errors
                hidden_errors = [sum(output_errors[i] * perceptron.weights[j] for i in range(self.output_size))
                                 for j, perceptron in enumerate(self.hidden_layer)]

                # Backpropagation - update weights and biases of hidden layer
                for i in range(self.hidden_size):
                    perceptron = self.hidden_layer[i]
                    perceptron.weights = [perceptron.weights[j] + learning_rate * hidden_errors[i] * input_data[j]
                                          for j in range(len(perceptron.weights))]
                    perceptron.bias += learning_rate * hidden_errors[i]


# Example usage
if __name__ == '__main__':
    # Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
    nn = NeuralNetwork(input_size=11, hidden_size=6, output_size=1)

    # Example dataset
    inputs = [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.5],
    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2],
    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ]

    # Example labels
    labels = [
        [0.7],
        [0.3],
        [0.5],
        [0.8],
        [0.2]
    ]

    # Training the neural network for 100 epochs
    nn.train(inputs, labels, learning_rate= 0.2, epochs=1000)

    # Testing the neural network
    for input_data in inputs:
        prediction = nn.forward(input_data)
        print("Prediction:", prediction)
