import torch
from train import train_model
from test import test_model
from sample_plt import display_random_samples
from data_loader import get_data_loaders
from model import *
import os

def main():
    # Define the name of the model being used.
    model_name = 'attention'

    # Define the directories for training and testing data.
    train_input_root_dir = 'data/DIV2K_train_HR/resized_25'
    train_output_root_dir = 'data/DIV2K_train_HR/resized_50'
    test_input_root_dir = 'data/DIV2K_valid_HR/resized_16'
    test_output_root_dir = 'data/DIV2K_valid_HR/resized_8'

    # Define the path to save the best trained model.
    best_model_path = 'saved_models/'+ model_name +'/best_' + model_name + '_image_enhancement_model.pth'

    # Define hyperparameters.
    batch_size = 100
    num_epochs = 200

    # Determine the computing device (GPU or CPU) available for training.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a directory to save trained models if it doesn't exist.
    model_save_dir = 'saved_models/' + model_name
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    # Create a directory to save test results if it doesn't exist.
    test_output_dir = 'results/' + model_name
    if not os.path.exists(test_output_dir):
        os.mkdir(test_output_dir)

    # Load training and testing data using data loaders.
    train_loader, test_loader = get_data_loaders(train_input_root_dir, train_output_root_dir, test_input_root_dir, test_output_root_dir, batch_size)

    # Initialize the model architecture.
    model = AttentionCNNImageEnhancementModel(input_channels=3, output_channels=3).to(device)

    # Train the model.
    train_model(model, train_loader, num_epochs, device, model_save_dir)

    # Test the model and save the best performing model.
    test_model(model, test_loader, device, test_output_dir, best_model_path)

    # Display random samples of test results.
    display_random_samples(test_output_dir, test_output_root_dir, num_samples=10)

if __name__ == "__main__":
    main()
