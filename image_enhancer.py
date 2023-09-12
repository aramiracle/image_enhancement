import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import math
from model import SimpleCNNImageEnhancementModel  # Import your custom model class

# Define the path to the latest saved model
model_path = 'saved_models/simple_cnn/best_cnn_image_enhancement_model.pth'

# Define the enhancement factor (log2(ratio))
enhancement_factor = 2.0  # Set the desired enhancement factor

# Create a transform to preprocess the input image
preprocess = transforms.Compose([
    transforms.ToTensor()  # Convert the input image to a PyTorch tensor
])

# Load the latest saved model
model = SimpleCNNImageEnhancementModel(input_channels=3, output_channels=3)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode (no gradient computation)

# Define the paths for input and output directories
input_directory = "image_enhancer/input"
output_directory = "image_enhancer/enhanced"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List all files in the input directory
input_files = os.listdir(input_directory)

# Calculate the number of enhancement iterations based on the enhancement factor
num_iterations = int(math.log2(enhancement_factor - 1e-8)) + 1

# Loop through each input image file
for input_file in input_files:
    # Construct the full path of the input image
    input_image_path = os.path.join(input_directory, input_file)

    # Load and preprocess the input image
    input_image = Image.open(input_image_path)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform image enhancement in multiple iterations
    with torch.no_grad():
        for _ in range(num_iterations):
            # Apply enhancement to the entire input image
            input_batch = model(input_batch)

    # Calculate the final resize factor
    final_resize_factor = enhancement_factor / (2 ** num_iterations)

    # Calculate the target dimensions based on the dimensions of input_batch
    target_width = int(input_batch.size(3) * final_resize_factor)
    target_height = int(input_batch.size(2) * final_resize_factor)

    # Resize the enhanced output to the calculated target dimensions
    output_image = transforms.functional.resize(input_batch.squeeze(0), (target_height, target_width))

    # Convert the resized output back to a PIL image
    output_image = transforms.ToPILImage()(output_image)

    # Construct the output image path
    output_file = os.path.splitext(input_file)[0] + "_enhanced.jpg"
    output_image_path = os.path.join(output_directory, output_file)

    # Save the enhanced and resized image
    output_image.save(output_image_path)

    print(f"Enhanced and resized image saved to {output_image_path}")
