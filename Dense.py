
    
import numpy as np
from Layer import Layer
from PIL import Image, ImageDraw, ImageFont

class Dense(Layer):
    
    def __init__(self, input_size, output_size, regularization=0.0):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)
        self.regularization = regularization
        self.num_neurons = output_size
        self.input_dim = input_size
        self.image = self.generate_image(output_size, img_text="Dense Layer")  # Pass the img_text parameter

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient):
        self.weights_gradient += np.dot(output_gradient, self.input.T) + 2 * self.regularization * self.weights
        self.bias_gradient += output_gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        return input_gradient 
    
    def update(self, learning_rate, batch_size):
        self.weights -= learning_rate * (self.weights_gradient / batch_size)
        self.bias -= learning_rate * (self.bias_gradient / batch_size)
        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)

    def generate_image(self, num_neurons, img_text="Dense Layer"):
        # Define image parameters
        image_width = 100
        image_height = 800  # Adjust the height as needed
        background_color = (0, 0, 0)
        neuron_color = (255, 255, 255)

        # Create a blank image
        image = Image.new("RGB", (image_width, image_height), background_color)
        draw = ImageDraw.Draw(image)

        # Calculate neuron positions
        neuron_spacing = image_height / (num_neurons + 1)
        neuron_radius = 5

        # Draw neurons and dots
        if num_neurons <= 10:
            for i in range(num_neurons):
                neuron_x = image_width // 2
                neuron_y = int(neuron_spacing * (i + 1))
                draw.ellipse((neuron_x - neuron_radius, neuron_y - neuron_radius,
                              neuron_x + neuron_radius, neuron_y + neuron_radius), fill=neuron_color)
        else:
            # Draw top 3 and bottom 3 neurons if there are more than 10 neurons
            for i in range(5):
                neuron_x = image_width // 2
                neuron_y = int(neuron_spacing * (i + 1))
                draw.ellipse((neuron_x - neuron_radius, neuron_y - neuron_radius,
                              neuron_x + neuron_radius, neuron_y + neuron_radius), fill=neuron_color)
                neuron_y = int(neuron_spacing * (num_neurons - i))
                draw.ellipse((neuron_x - neuron_radius, neuron_y - neuron_radius,
                              neuron_x + neuron_radius, neuron_y + neuron_radius), fill=neuron_color)

            # Draw a text in the middle indicating the total number of neurons
            middle_x = image_width // 2
            middle_y = image_height // 2
            font = ImageFont.load_default()
            text = f"Total: {num_neurons}"
            text_size = font.getbbox(text)
            text_x = middle_x - text_size[0] - text_size[2] // 2
            text_y = middle_y - text_size[1] // 2
            draw.text((text_x, text_y), text, fill=neuron_color, font=font)

        # Add the input text above the image
        font = ImageFont.load_default()
        text_size = font.getbbox(img_text)
        text_x = image_width // 2 - text_size[0] - text_size[2] // 2
        text_y = 10  # Adjust the vertical position as  needed
        draw.text((text_x, text_y), img_text, fill=neuron_color, font=font)

        return image




