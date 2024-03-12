import math

class Perceptron:
    def __init__(self, num_inputs, init_weight, init_bias = 0, activation_fcn='tanh'):
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
        elif self.activation_fcn == 'nofn':
            return self.nofn(weighted_sum)
        else:
            raise ValueError("Wrong Activation Function Input!!")

    # -------------------- Activation Functions ------------------
    # No function
    def nofn(self, x):
        return x

    # Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Tanh
    def tanh(self, x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    def relu(self, x):
        return max(0, x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size,output_size,activation_function):
        self.input_size = input_size
        self.hidden_layer1_size = hidden_layer1_size
        self.hidden_layer2_size = hidden_layer2_size
        self.output_size = output_size

        # Create hidden layer 1 perceptrons
        self.hidden_layer_1 = [Perceptron(num_inputs=input_size, init_weight=0.1, init_bias=0,activation_fcn = activation_function ) for _ in range(hidden_layer1_size)]

        # Create hidden layer 2 perceptrons
        self.hidden_layer_2 = [Perceptron(num_inputs=hidden_layer1_size, init_weight=0.1, init_bias=0,activation_fcn = activation_function) for _ in range(hidden_layer2_size)]

        # Create output layer perceptrons
        self.output_layer = [Perceptron(num_inputs=hidden_layer2_size, init_weight=0.1, init_bias=0,activation_fcn='nofn') for _ in range(output_size)]

    def forward(self, inputs):
        # Forward pass through the neural network
        hidden_layer_1_outputs = [perceptron.predict(inputs) for perceptron in self.hidden_layer_1]
        hidden_layer_2_outputs = [perceptron.predict(hidden_layer_1_outputs) for perceptron in self.hidden_layer_2]
        output_layer_outputs = [perceptron.predict(hidden_layer_2_outputs) for perceptron in self.output_layer]
        return output_layer_outputs

    def train(self, inputs, labels, learning_rate=0.1, epochs=1):
        for _ in range(epochs):
            for input_data, label in zip(inputs, labels):
                # Forward pass
                hidden_layer_1_outputs = [perceptron.predict(input_data) for perceptron in self.hidden_layer_1]
                hidden_layer_2_outputs = [perceptron.predict(hidden_layer_1_outputs) for perceptron in self.hidden_layer_2]
                output_layer_outputs = [perceptron.predict(hidden_layer_2_outputs) for perceptron in self.output_layer]

                # Calculate output layer deltas
                # Delta = (t-o)o(1-o)
                # t = label, o = output from output layer
                output_deltas = [(label[i] - output_layer_outputs[i])*output_layer_outputs[i]*(1 -  output_layer_outputs[i]) for i in range(self.output_size)]

                # Backpropagation - update weights and biases of output layer
                for i in range(self.output_size):
                    perceptron = self.output_layer[i]
                    perceptron.weights = [perceptron.weights[j] + learning_rate * output_deltas[i] * hidden_layer_2_outputs[j]
                                          for j in range(len(perceptron.weights))]
                    # Commenting out Updating bias for now
                    # perceptron.bias += learning_rate * output_deltas[i]
                
                # Calculate hidden layer 2 deltas
                # Delta = o(1-o) * sum(output_delta*output_weights)
                # o = output from hidden layer 2
                # Calculate o(1-o) first --> preparing for hidden 2 deltas
                pre_hidden_2_deltas = [hidden_layer_2_outputs[i]*(1 -  hidden_layer_2_outputs[i]) for i in range(self.hidden_layer2_size)]
                # Backpropagation for hidden layer 2
                for i, perceptron in enumerate(self.hidden_layer_2):
                    # Calculate sum(delta.output * weights.output)
                    downstream_sum = sum(output_deltas[j] * self.output_layer[j].weights[i] for j in range(self.output_size))
                    hidden2_deltas = [downstream_sum * element for element in pre_hidden_2_deltas]
                    # Update weights of hidden layer 2
                    # delta_weights = learning rate * delta_hidden * hidden_1_output
                    perceptron.weights = [perceptron.weights[j] + learning_rate * hidden2_deltas[i] * hidden_layer_1_outputs[j]
                                          for j in range(len(perceptron.weights))]
                    # perceptron.bias += learning_rate * hidden_error

                # Calculate hidden layer 1 deltas
                # Delta = o(1-o) * sum(hidden2_delta*hidden2_weights)
                # o = output from hidden layer 1
                # Calculate o(1-o) first --> preparing for hidden 1 deltas
                pre_hidden_1_deltas = [hidden_layer_1_outputs[i]*(1 -  hidden_layer_1_outputs[i]) for i in range(self.hidden_layer1_size)]
                # Backpropagation for hidden layer 2
                for i, perceptron in enumerate(self.hidden_layer_1):
                    # Calculate sum(delta.output * weights.output)
                    downstream_sum = sum(hidden2_deltas[j] * self.hidden_layer_2[j].weights[i] for j in range(self.hidden_layer2_size))
                    hidden1_deltas = [downstream_sum * element for element in pre_hidden_1_deltas]
                    # Update weights of hidden layer 2
                    # delta_weights = learning rate * delta_hidden * hidden_1_output
                    perceptron.weights = [perceptron.weights[j] + learning_rate * hidden1_deltas[i] * input_data[j]
                                          for j in range(len(perceptron.weights))]
                    # perceptron.bias += learning_rate * hidden_error

# Example usage
if __name__ == '__main__':
    # Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
    nn = NeuralNetwork(input_size = 11, hidden_layer1_size = 5, hidden_layer2_size = 3, output_size = 1, activation_function = 'relu')

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
    nn.train(inputs, labels, learning_rate= 2, epochs=1000)

    # Testing the neural network
    for input_data in inputs:
        prediction = nn.forward(input_data)
        print("Prediction:", prediction)
