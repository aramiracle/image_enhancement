import torch
from train import train_model
from test import test_model
from sample_plt import display_random_samples
from data_loader import get_data_loaders
from model import *
import os
import shutil

def main():
    # Define the name of the model being used.
    model_name = 'simple_residual_ssim_psnr_lnl1'

    # Define the directories for training and testing data.
    train_input_root_dir = 'data/DIV2K_train_HR/resized_25'
    train_output_root_dir = 'data/DIV2K_train_HR/resized_50'
    test_input_root_dir = 'data/DIV2K_valid_HR/resized_8'
    test_output_root_dir = 'data/DIV2K_valid_HR/resized_4'
    
    # Define the directory for saving generated images during testing.
    generated_images_dir = 'results/generated_images'

    # Define the path to save the best trained model.
    best_model_path = f'saved_models/{model_name}/best_cnn_image_enhancement_model.pth'

    # Define hyperparameters.
    batch_size = 32
    num_epochs = 1000

    # Determine the computing device (GPU or CPU) available for training.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a directory to save trained models if it doesn't exist.
    model_save_dir = 'saved_models/' + model_name
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)

    # Create a directory to save test results if it doesn't exist.
    test_output_dir = 'results/' + model_name
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir, exist_ok=True)

    # Clear the existing generated images directory and create it.
    shutil.rmtree(generated_images_dir, ignore_errors=True)
    os.makedirs(generated_images_dir, exist_ok=True)

    # Load training and testing data using data loaders.
    train_loader, test_loader = get_data_loaders(train_input_root_dir, train_output_root_dir, test_input_root_dir, test_output_root_dir, batch_size)

    # Initialize the model architecture.
    model = Generator().to(device)

    # Train the model.
    train_model(model, train_loader, num_epochs, device, model_save_dir, generated_images_dir, criterion_str='SSIM_PSNR_LNL1')

    # Test the model and save the best performing model.
    test_model(model, test_loader, device, test_output_dir, best_model_path)

    # Display random samples of test results.
    display_random_samples(test_output_dir, test_output_root_dir, num_samples=10)

if __name__ == "__main__":
    main()
