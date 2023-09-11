import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import multiprocessing

# Define the input folders for training and test datasets
train_input_folder = 'data/DIV2K_train_HR/original'
test_input_folder = 'data/DIV2K_valid_HR/original'
train_save_dir = 'data/DIV2K_train_HR'
test_save_dir = 'data/DIV2K_valid_HR'

# Define the target sizes for training data
target_sizes = [25, 50, 100, 200, 400]

# Define the fractions for test data
fractions = [1/2, 1/4, 1/8, 1/16, 1/32]

def resize_and_save_images(image_file, input_folder, output_folder, target_size=None, fraction=None):
    # Load the image using PIL
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path)

    # Resize the image based on the specified target size or fraction
    if target_size:
        resize_transform = transforms.Resize((target_size, target_size))
        resized_image = resize_transform(image)
    elif fraction:
        original_width, original_height = image.size
        min_width = int(original_width * min(fractions))
        min_height = int(original_height * min(fractions))
        new_width = int(min_width * (fraction // min(fractions)))
        new_height = int(min_height * (fraction // min(fractions)))
        resize_transform = transforms.Resize((new_height, new_width))
        resized_image = resize_transform(image)

    # Save the resized image in the corresponding output folder
    output_path = os.path.join(output_folder, image_file)
    resized_image.save(output_path)

if __name__ == "__main__":
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Use the number of CPU cores

    # Process the training dataset and save in 'data/DIV2K_train_HR'
    for target_size in target_sizes:
        train_output_folder = os.path.join(train_save_dir, f'resized_{target_size}')
        os.makedirs(train_output_folder, exist_ok=True)

        train_image_files = [f for f in os.listdir(train_input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        train_args = [(image_file, train_input_folder, train_output_folder, target_size, None) for image_file in train_image_files]

        # Use tqdm with multiprocessing to resize and save training images
        with tqdm(total=len(train_args), desc=f"Processing Training (Size {target_size})") as pbar:
            for _ in pool.starmap(resize_and_save_images, train_args):
                pbar.update(1)

    # Process the test dataset and save in 'data/DIV2K_valid_HR'
    for fraction in fractions:
        test_output_folder = os.path.join(test_save_dir, f'resized_{int(1/fraction)}')
        os.makedirs(test_output_folder, exist_ok=True)

        test_image_files = [f for f in os.listdir(test_input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        test_args = [(image_file, test_input_folder, test_output_folder, None, fraction) for image_file in test_image_files]

        # Use tqdm with multiprocessing to resize and save test images
        with tqdm(total=len(test_args), desc=f"Processing Test (Fraction {int(1/fraction)})") as pbar:
            for _ in pool.starmap(resize_and_save_images, test_args):
                pbar.update(1)

    # Close the pool of worker processes
    pool.close()
    pool.join()

    print("Image resizing and saving completed.")
