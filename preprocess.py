import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Define the input folders for training and test datasets
train_input_folder = 'data/DIV2K_train_HR/original'
test_input_folder = 'data/DIV2K_valid_HR/original'
train_save_dir = 'data/DIV2K_train_HR'
test_save_dir = 'data/DIV2K_valid_HR'

# Define the target sizes for training data
target_sizes = [25, 50, 100, 200, 400]

# Define the fractions for test data
fractions = [1/4, 1/8, 1/16, 1/32]

# Process the training dataset and save in 'data/DIV2K_train_HR'
for target_size in target_sizes:
    # Create output folder within 'train_save_dir'
    train_output_folder = os.path.join(train_save_dir, f'resized_{target_size}')
    os.makedirs(train_output_folder, exist_ok=True)

    # List only image files in the training input folder (filter by extension)
    train_image_files = [f for f in os.listdir(train_input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Initialize the progress bar
    train_progress_bar = tqdm(total=len(train_image_files), desc=f"Processing Training (Size {target_size})")

    for image_file in train_image_files:
        # Load the image using PIL
        image_path = os.path.join(train_input_folder, image_file)
        image = Image.open(image_path)

        # Resize the image to the target size for training data
        resize_transform = transforms.Resize((target_size, target_size))
        resized_image = resize_transform(image)

        # Save the resized image in the corresponding output folder
        output_path = os.path.join(train_output_folder, image_file)
        resized_image.save(output_path)

        # Update the progress bar for training data
        train_progress_bar.update(1)

    # Close the progress bar for training data
    train_progress_bar.close()

# Process the test dataset and save in 'data/DIV2K_valid_HR'
for fraction in fractions:
    # Create output folder within 'test_save_dir'
    test_output_folder = os.path.join(test_save_dir, f'resized_{int(1/fraction)}')
    os.makedirs(test_output_folder, exist_ok=True)

    # List only image files in the test input folder (filter by extension)
    test_image_files = [f for f in os.listdir(test_input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Initialize the progress bar for test data
    test_progress_bar = tqdm(total=len(test_image_files), desc=f"Processing Test (Fraction {int(1/fraction)})")

    for image_file in test_image_files:
        # Load the image using PIL
        image_path = os.path.join(test_input_folder, image_file)
        image = Image.open(image_path)

        # Calculate new dimensions while maintaining the aspect ratio
        original_width, original_height = image.size
        new_width = int(original_width * fraction)
        new_height = int(original_height * fraction)

        resize_transform = transforms.Resize((new_height, new_width))
        resized_image = resize_transform(image)

        # Save the resized image in the corresponding output folder
        output_path = os.path.join(test_output_folder, image_file)
        resized_image.save(output_path)

        # Update the progress bar for test data
        test_progress_bar.update(1)

    # Close the progress bar for test data
    test_progress_bar.close()

print("Image resizing and saving completed.")
