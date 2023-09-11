import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def display_random_samples(test_output_dir, test_output_root_dir, num_samples=10):
    # List the files in the test_output_dir that start with "enhanced_"
    enhanced_files = [f for f in os.listdir(test_output_dir) if f.startswith("enhanced_")]

    # Shuffle the list of enhanced files
    random.shuffle(enhanced_files)

    # Select num_samples random enhanced images
    random_samples = enhanced_files[:num_samples]

    # Display the selected random enhanced images along with random real images
    for i, enhanced_file in enumerate(random_samples):
        enhanced_image = Image.open(os.path.join(test_output_dir, enhanced_file))
        
        # Find a random real image
        real_files = os.listdir(test_output_root_dir)
        random_real_file = random.choice(real_files)
        real_image = Image.open(os.path.join(test_output_root_dir, random_real_file))

        # Display the images using matplotlib
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(enhanced_image)
        plt.axis('off')
        plt.title("Enhanced")

        plt.subplot(2, num_samples, i + num_samples + 1)
        plt.imshow(real_image)
        plt.axis('off')
        plt.title("Real")

    plt.show()