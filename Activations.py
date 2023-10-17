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


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)
        self.image = self.generate_image(img_text="Tanh Layer")  # Add this line to generate an image

    def generate_image(self, img_text="Tanh Layer"):
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

        # Draw the activation function symbol (you can customize this)
        font = ImageFont.load_default()
        activation_text = "Tanh"  # Customize the text if needed
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

class Linear(Activation):
    def __init__(self):
        def linear(x):
            return x

        def linear_prime(x):
            return 1

        super().__init__(linear, linear_prime)
        


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)
        