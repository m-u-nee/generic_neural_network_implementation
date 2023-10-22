import numpy as np
from scipy import signal
from Layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        self.kernels_gradient = np.zeros(self.kernels_shape)
        self.biases_gradient = np.zeros(self.output_shape)
        self.input_gradient = np.zeros(self.input_shape)
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient):
        

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.kernels_gradient[i, j] += signal.correlate2d(self.input[j], output_gradient[i], "valid")
                self.input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        self.biases_gradient += output_gradient
        
        return self.input_gradient
    
    def update(self,learning_rate,batch_size):
        self.kernels -= learning_rate * self.kernels_gradient/batch_size
        self.biases -= learning_rate * self.biases_gradient/batch_size
        self.kernels_gradient = np.zeros(self.kernels_shape)
        self.input_gradient = np.zeros(self.input_shape)
        
