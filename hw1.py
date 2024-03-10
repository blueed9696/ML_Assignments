class Perceptron:
    def __init__(self, num_inputs, init_weight, init_bias, activation_fcn = 'sigmoid'):
        # Init Weights (All to 0.1)
        self.weights = [init_weight for _ in range(num_inputs)]
        # Init Bias
        self.bias = init_bias
        # Init activation fcn
        self.actiavtion_fcn = activation_fcn
    
    def predict(self, inputs):
        # Weighted Sum
        weighted_sum = sum(inputs[i]*self.weights[i] for i in range(len(inputs))) + self.bias
        
        # Apply activation function
        if self.actiavtion_fcn == 'sigmoid':
            return self.sigmoid(weighted_sum)
        elif self.actiavtion_fcn == 'tanh':
            return self.tanh(weighted_sum)
        elif self.actiavtion_fcn == 'relu':
            return self.relu(weighted_sum)
        else:
            raise ValueError("Wrong Activation Function Input!!")
    
    # -------------------- Activation Functions ------------------
    # Exponential expanded with taylor series
    def exponential(self, x):
        return 1 + x + (x ** 2) / 2 + (x ** 3) / 6 + (x ** 4) / 24 + (x ** 5) / 120

    # Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + self.exp(-x))
    
    # Tanh
    def tanh(self, x):
        return (self.exp(x) - self.exp(-x)) / (self.exp(x) + self.exp(-x))

    def relu(self, x):
        return max(0, x)
    
    def train(self, inputs, label, learning_rate = 0.1):
        # Make a prediction
        prediction = self.predict(inputs)
        # Update weights and bias based an error
        error = label - prediction
        self.weights = [self.weights[i] + learning_rate * error * inputs[i] for i in range(len(inputs))]
        self.bias += learning_rate * error
# Main
if __name__ == '__main__':
    # Create Perceptron Constructor
    # (num_perceptrons, init_weight, init_bias, activation_fcn = 'sigmoid'|'tanh'|'relu')
    perceptron = Perceptron(2, 0.1, 0)



