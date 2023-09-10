import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNImageEnhancementModel  # Import your custom model class
import math

# Define the path to the latest saved model
model_path = "saved_models/cnn_image_enhancement_model_final.pth"

# Define the enhancement factor (log2(ratio))
enhancement_factor = 5.0  # Set the enhancement factor to 5

# Create a transform to preprocess the input image
preprocess = transforms.Compose([
    transforms.ToTensor()  # Convert to a PyTorch tensor
])

# Load the latest saved model
model = CNNImageEnhancementModel(input_channels=3, output_channels=3)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode (no gradient computation)

# Load and preprocess the new image
input_image_path = "single_test/image.jpg"  # Replace with the path to your input image
input_image = Image.open(input_image_path)
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Calculate the number of enhancement iterations based on the enhancement factor
num_iterations = int(math.log2(enhancement_factor))

# Perform image enhancement in multiple iterations
with torch.no_grad():
    for _ in range(num_iterations):
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

# Save the enhanced and resized image
output_image_path = "single_test/output_image.jpg"  # Specify the output image path
output_image.save(output_image_path)

print(f"Enhanced and resized image saved to {output_image_path}")
