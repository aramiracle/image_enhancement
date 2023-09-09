import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Define the input and output folders
input_folder = 'data/DIV2K_train_HR'
output_folder_1_16 = 'data/DIV2K_train_HR_1_16'  # Add a folder for 1/16 ratio images
output_folder_1_8 = 'data/DIV2K_train_HR_1_8'
output_folder_1_4 = 'data/DIV2K_train_HR_1_4'

# Create output folders if they don't exist
os.makedirs(output_folder_1_16, exist_ok=True)  # Create the 1/16 folder
os.makedirs(output_folder_1_8, exist_ok=True)
os.makedirs(output_folder_1_4, exist_ok=True)


# Define the resizing transformations
sample_image_path = os.path.join(input_folder, os.listdir(input_folder)[0])
sample_image = Image.open(sample_image_path)
height, width = sample_image.size

resize_1_16 = transforms.Resize((height // 16, width // 16))  # Add a transformation for 1/16
resize_1_8 = transforms.Resize((height // 8, width // 8))
resize_1_4 = transforms.Resize((height // 4, width // 4))

# List all files in the input folder
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# Initialize the progress bar
progress_bar = tqdm(total=len(image_files), desc="Processing")

for image_file in image_files:
    # Load the image using PIL
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path)

    # Resize the image to 1/8 ratio and save it
    resized_image_1_8 = resize_1_8(image)
    output_path_1_8 = os.path.join(output_folder_1_8, image_file)
    resized_image_1_8.save(output_path_1_8)

    # Resize the image to 1/4 ratio and save it
    resized_image_1_4 = resize_1_4(image)
    output_path_1_4 = os.path.join(output_folder_1_4, image_file)
    resized_image_1_4.save(output_path_1_4)

    # Resize the image to 1/16 ratio and save it
    resized_image_1_16 = resize_1_16(image)
    output_path_1_16 = os.path.join(output_folder_1_16, image_file)
    resized_image_1_16.save(output_path_1_16)

    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

print("Image resizing and saving completed.")
