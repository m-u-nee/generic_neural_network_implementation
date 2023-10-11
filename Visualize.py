import numpy as np
from PIL import Image, ImageDraw, ImageFont
from Activations import Tanh
from Dense import Dense

def visualize_network(layers):
    total_width = sum(layer.image.width for layer in layers)
    max_height = max(layer.image.height for layer in layers)

    # Create a blank image to hold the entire network visualization
    network_image = Image.new("RGB", (total_width, max_height), (0, 0, 0))
    x_offset = 0

    for layer in layers:
        network_image.paste(layer.image, (x_offset, (max_height - layer.image.height) // 2))
        x_offset += layer.image.width
        # Add white line to separate layers
        draw = ImageDraw.Draw(network_image)
        draw.line((x_offset, 0, x_offset, max_height), fill=(255, 255, 255), width=1)
        x_offset += 1


    

    return network_image


