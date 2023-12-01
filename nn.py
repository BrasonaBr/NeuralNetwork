import numpy as np
import json

class NeuralNetwork:
    def __init__(self, input_size = 1, hidden_size = 1, output_size = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialização dos pesos e vieses
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def save(self, filename="trained_nn"):
        weights_data = {
            "weights_input_hidden": self.weights_input_hidden.tolist(),
            "bias_hidden": self.bias_hidden.tolist(),
            "weights_hidden_output": self.weights_hidden_output.tolist(),
            "bias_output": self.bias_output.tolist(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "activation_function": self.activation.__name__,
            "gradient_function": self.gradient.__name__,
            "output_function": self.output_function.__name__
        }

        with open(f"{filename}.json", 'w') as file:
            json.dump(weights_data, file)

    def load(self, filename="trained_nn"):

        with open(f"{filename}.json", 'r') as file:
            weights_data = json.load(file)

        self.weights_input_hidden = np.array(weights_data["weights_input_hidden"])
        self.bias_hidden = np.array(weights_data["bias_hidden"])
        self.weights_hidden_output = np.array(weights_data["weights_hidden_output"])
        self.bias_output = np.array(weights_data["bias_output"])
        self.input_size = weights_data["input_size"]
        self.hidden_size = weights_data["hidden_size"]
        self.output_size = weights_data["output_size"]

        self.activation = getattr(self, weights_data["activation_function"])
        self.gradient = getattr(self, weights_data["gradient_function"])
        self.output_function = getattr(self, weights_data["output_function"])

    #Função de ativação
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #Derivada função de ativação (gradiente)
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def softmax_derivative(self, x):
        s = self.softmax(x)
        jacobian_matrix = np.diag(s) - np.outer(s, s)
        return jacobian_matrix
    
    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return np.ones_like(x)
    
    activation = sigmoid
    gradient = sigmoid_derivative
    output_function = sigmoid

    def set_output_function(self, function):
        output_functions = {
            "sigmoid": self.sigmoid,
            "relu": self.relu,
            "tanh": self.tanh,
            "linear": self.linear,
            "softmax": self.softmax
        }

        if function in output_functions:
            self.output_function = output_functions[function]
        else:
            raise ValueError(f"Função de ativação '{function}' não suportada.")

    def set_activation(self, function):
        activation_functions = {
            "sigmoid": (self.sigmoid, self.sigmoid_derivative),
            "relu": (self.relu, self.relu_derivative),
            "tanh": (self.tanh, self.tanh_derivative),
            "linear": (self.linear, self.linear_derivative),
            "softmax": (self.softmax, self.softmax_derivative)
        }

        if function in activation_functions:
            self.activation, self.gradient = activation_functions[function]
        else:
            raise ValueError(f"Função de ativação '{function}' não suportada.")

    def forward(self, input_data):
        # Passagem direta pela rede
        self.hidden_input = np.dot(input_data, self.weights_input_hidden)
        self.hidden_output = self.activation(self.hidden_input + self.bias_hidden)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        # Change the activation function in the output layer
        self.predicted_output = self.output_function(self.final_input + self.bias_output)

        return self.predicted_output.tolist()

    def backward(self, input_data, target, learning_rate):
        # Cálculo do erro
        error = target - self.predicted_output

        # Retropropagação do erro
        output_error = error * self.gradient(self.predicted_output)
        hidden_layer_error = output_error.dot(self.weights_hidden_output.T) * self.gradient(self.hidden_output)

        # Atualização dos pesos e vieses
        self.weights_hidden_output += self.hidden_output.T.dot(output_error) * learning_rate
        self.bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden += input_data.T.dot(hidden_layer_error) * learning_rate
        self.bias_hidden += np.sum(hidden_layer_error, axis=0, keepdims=True) * learning_rate

    def train(self, train_data, target_data, epochs, learning_rate):
        train_data = np.array(train_data)
        target_data = np.array(target_data)

        for epoch in range(epochs):
            # Passagem direta
            self.forward(train_data)

            # Retropropagação e atualização dos pesos
            self.backward(train_data, target_data, learning_rate)

            # Cálculo do erro médio
            error = np.mean(np.abs(target_data - self.predicted_output))

        return error
