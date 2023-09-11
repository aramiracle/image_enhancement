import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def display_random_samples(test_output_dir, test_output_root_dir, num_samples=1, figsize=(8, 8)):
    # Generate random numbers between 0 and the number of test images - 1
    random_numbers = random.sample(range(len(test_output_dir)), num_samples)

    for i, random_number in enumerate(random_numbers):
        # Get the corresponding index and filenames
        index = random_number
        enhanced_file = sorted(os.listdir(test_output_dir))[index]
        real_file = sorted(os.listdir(test_output_root_dir))[index]

        # Load the enhanced and real images
        enhanced_image = Image.open(os.path.join(test_output_dir, enhanced_file))
        real_image = Image.open(os.path.join(test_output_root_dir, real_file))

        # Create a figure with subplots for displaying the image pair
        plt.figure(figsize=figsize)

        # Display the enhanced image in the first subplot
        plt.subplot(1, 2, 1)
        plt.imshow(enhanced_image)
        plt.axis('off')
        plt.title("Enhanced")

        # Display the real image in the second subplot
        plt.subplot(1, 2, 2)
        plt.imshow(real_image)
        plt.axis('off')
        plt.title("Real")

        # Show the figure for this pair of images
        plt.show()