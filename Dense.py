import numpy as np
from Layer import Layer
class Dense(Layer):
    
    def __init__(self, input_size, output_size, regularization = 0.0):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)
        self.regularization = regularization
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient):

        self.weights_gradient += np.dot(output_gradient,self.input.T) + 2 * self.regularization * self.weights
        self.bias_gradient += output_gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        return input_gradient
    
    def update(self, learning_rate,batch_size):
        self.weights -= learning_rate*(self.weights_gradient/batch_size)
        self.bias -= learning_rate*(self.bias_gradient/batch_size)
        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)