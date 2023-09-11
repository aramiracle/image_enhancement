import os
import torch
import torchvision.transforms as transforms
import math
from model import SimpleCNNImageEnhancementModel  # Import your custom model class
import imageio

# Define the path to the latest saved model
model_path = "saved_models/simple_cnn/best_cnn_image_enhancement_model.pth"

# Define the enhancement factor (log2(ratio))
enhancement_factor = 3.0  # Set the enhancement factor

# Create a transform to preprocess the input image
preprocess = transforms.Compose([
    transforms.ToTensor()  # Convert to a PyTorch tensor
])

# Load the latest saved model
model = SimpleCNNImageEnhancementModel(input_channels=3, output_channels=3)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode (no gradient computation)

# Define the paths for input and output directories
input_directory = "gif_enhancer/input"
output_directory = "gif_enhancer/enhanced"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List all files in the input directory
input_files = os.listdir(input_directory)

# Calculate the number of enhancement iterations based on the enhancement factor
num_iterations = int(math.log2(enhancement_factor - 1e-8)) + 1

# Loop through each input GIF file
for input_file in input_files:
    # Construct the full path of the input GIF
    input_gif_path = os.path.join(input_directory, input_file)

    # Load the input GIF
    input_gif = imageio.mimread(input_gif_path)

    enhanced_frames = []

    # Loop through each frame in the GIF
    for frame in input_gif:
        # Convert the frame to a PyTorch tensor
        frame_tensor = transforms.ToTensor()(frame)
        frame_tensor = frame_tensor.unsqueeze(0)  # Add a batch dimension

        # Perform image enhancement in multiple iterations
        with torch.no_grad():
            for _ in range(num_iterations):
                # Enhance the entire frame
                frame_tensor = model(frame_tensor)

        # Calculate the final resize factor
        final_resize_factor = enhancement_factor / (2 ** num_iterations)

        # Calculate the target dimensions based on the dimensions of frame_tensor
        target_width = int(frame_tensor.size(3) * final_resize_factor)
        target_height = int(frame_tensor.size(2) * final_resize_factor)

        # Resize the enhanced output to the calculated target dimensions
        enhanced_frame = transforms.functional.resize(frame_tensor.squeeze(0), (target_height, target_width))

        # Convert the resized output back to a NumPy array (image)
        enhanced_frame = transforms.ToPILImage()(enhanced_frame)
    
        # Append the enhanced frame to the list
        enhanced_frames.append(enhanced_frame)

    # Construct the output GIF path
    output_gif_file = os.path.splitext(input_file)[0] + "_enhanced.gif"
    output_gif_path = os.path.join(output_directory, output_gif_file)

    # Save the enhanced GIF
    imageio.mimsave(output_gif_path, enhanced_frames, loop=0)

    print(f"Enhanced GIF saved to {output_gif_path}")
