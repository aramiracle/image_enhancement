import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

input_folders = ['data/DIV2K_train_HR', 'data/DIV2K_valid_HR']
# Define the input and output folders
for input_folder in input_folders:
    output_folder_1 = input_folder + '_800x640'
    output_folder_2 = input_folder + '_400x320'
    output_folder_3 = input_folder + '_200x160'
    output_folder_4 = input_folder + '_100x80'
    output_folder_5 = input_folder + '_50x40'
    

    # Create output folders if they don't exist
    os.makedirs(output_folder_1, exist_ok=True)
    os.makedirs(output_folder_2, exist_ok=True)
    os.makedirs(output_folder_3, exist_ok=True)
    os.makedirs(output_folder_4, exist_ok=True)
    os.makedirs(output_folder_5, exist_ok=True)

    # Define the resizing transformations
    sample_image_path = os.path.join(input_folder, os.listdir(input_folder)[0])
    sample_image = Image.open(sample_image_path)
    height, width = sample_image.size

    resize = transforms.Resize((800, 640))
    resize_1_2 = transforms.Resize((400, 320))
    resize_1_4 = transforms.Resize((200, 160))
    resize_1_8 = transforms.Resize((100, 80))
    resize_1_16 = transforms.Resize((50, 40))

    # List all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Initialize the progress bar
    progress_bar = tqdm(total=len(image_files), desc="Processing")

    for image_file in image_files:
        # Load the image using PIL
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)

        # Resize the image and save it
        resized_image = resize(image)
        output_path = os.path.join(output_folder_1, image_file)
        resized_image.save(output_path)

        # Resize the image to 1/2 ratio and save it
        resized_image_1_2 = resize_1_2(image)
        output_path = os.path.join(output_folder_2, image_file)
        resized_image_1_2.save(output_path)

        # Resize the image to 1/4 ratio and save it
        resized_image_1_4 = resize_1_4(image)
        output_path = os.path.join(output_folder_3, image_file)
        resized_image_1_4.save(output_path)

        # Resize the image to 1/8 ratio and save it
        resized_image_1_8 = resize_1_8(image)
        output_path = os.path.join(output_folder_4, image_file)
        resized_image_1_8.save(output_path)

        # Resize the image to 1/16 ratio and save it
        resized_image_1_16 = resize_1_16(image)
        output_path = os.path.join(output_folder_5, image_file)
        resized_image_1_16.save(output_path)

        # Update the progress bar
        progress_bar.update(1)

# Close the progress bar
progress_bar.close()

print("Image resizing and saving completed.")
