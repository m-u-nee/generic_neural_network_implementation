from Layer import Layer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
class Activation(Layer):

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self,output_gradient):
        return np.multiply(output_gradient, self.activation_prime(self.input))

    def generate_image(self, activation_text, img_text):
        # Define image parameters
        image_width = 100
        image_height = 800
        background_color = (0, 0, 0)
        activation_color = (255, 255, 255)

        # Create a blank image
        image = Image.new("RGB", (image_width, image_height), background_color)
        draw = ImageDraw.Draw(image)

        # Calculate the position of the activation function symbol
        activation_x = image_width // 2
        activation_y = image_height // 2

        # Draw the activation function symbol
        font = ImageFont.load_default()
        text_size = font.getbbox(activation_text)
        text_x = activation_x - text_size[0] - text_size[2] // 2
        text_y = activation_y - text_size[1] // 2
        draw.text((text_x, text_y), activation_text, fill=activation_color, font=font)

        # Add the input text above the image
        text_size = font.getbbox(img_text)
        text_x = image_width // 2 - text_size[0] - text_size[2] // 2
        text_y = 10  # Adjust the vertical position as needed
        draw.text((text_x, text_y), img_text, fill=activation_color, font=font)

        return image


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)
        self.image = self.generate_image("Tanh", "Tanh Layer")

class Linear(Activation):
    def __init__(self):
        def linear(x):
            return x

        def linear_prime(x):
            return 1

        super().__init__(linear, linear_prime)
        self.image = self.generate_image("Linear", "Linear Layer")
        


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
        self.image = self.generate_image("Sigmoid", "Sigmoid Layer")

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
    
    
        
class ReLu(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)
        self.image = self.generate_image("ReLU", "ReLU Layer")        
    
class Binary(Activation):
    def __init__(self):
        def binary(x):
            return np.where(x > 0, 1, 0)

        def binary_prime(x):
            return np.where(x > 0, 1, 0)

        super().__init__(binary, binary_prime)
        self.image = self.generate_image("Binary", "Binary Layer")

class LeakyReLU(Activation):
    def __init__(self):
        def leaky_relu(x):
            return np.where(x > 0, x, x * 0.01)

        def leaky_relu_prime(x):
            return np.where(x > 0, 1, 0.01)

        super().__init__(leaky_relu, leaky_relu_prime)
        self.image = self.generate_image("Leaky ReLU", "Leaky ReLU Layer")


class ELU(Activation):
    def __init__(self):
        def elu(x):
            return np.where(x > 0, x, np.exp(x) - 1)

        def elu_prime(x):
            return np.where(x > 0, 1, np.exp(x))

        super().__init__(elu, elu_prime)
        self.image = self.generate_image("ELU", "ELU Layer")

class SELU(Activation):
    def __init__(self):
        def selu(x):
            return np.where(x > 0, 1.0507 * x, 1.0507 * 1.67326 * (np.exp(x) - 1))

        def selu_prime(x):
            return np.where(x > 0, 1.0507, 1.0507 * 1.67326 * np.exp(x))

        super().__init__(selu, selu_prime)
        self.image = self.generate_image("SELU", "SELU Layer")




    