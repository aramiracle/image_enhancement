import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNImageEnhancementModel  # Import your custom model class

# Define the path to the latest saved model
model_path = "saved_models/cnn_image_enhancement_model_final.pth"

# Create a transform to preprocess the input image
preprocess = transforms.Compose([
    transforms.ToTensor()        # Convert to a PyTorch tensor
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

# Perform image enhancement
with torch.no_grad():
    enhanced_output = model(input_batch)

# Convert the enhanced output back to a PIL image
output_image = transforms.ToPILImage()(enhanced_output.squeeze(0))

# Save the enhanced image
output_image_path = "single_test/output_image.jpg"  # Specify the output image path
output_image.save(output_image_path)

print(f"Enhanced image saved to {output_image_path}")
