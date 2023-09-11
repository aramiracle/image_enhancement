import torch
from train import train_model
from test import test_model
from data_loader import get_data_loaders
from model import *
import os

def main():
    train_input_root_dir = 'data/DIV2K_train_HR/resized_50'
    train_output_root_dir = 'data/DIV2K_train_HR/resized_100'
    test_input_root_dir = 'data/DIV2K_valid_HR/resized_4'
    batch_size = 100
    num_epochs = 120
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_save_dir = 'saved_models/cnn'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    test_output_dir = 'results/cnn'
    if not os.path.exists(test_output_dir):
        os.mkdir(test_output_dir)

    train_loader, test_loader = get_data_loaders(train_input_root_dir, train_output_root_dir, test_input_root_dir, batch_size)

    model = CNNImageEnhancementModel(input_channels=3, output_channels=3).to(device)

    train_model(model, train_loader, num_epochs, device, model_save_dir)

    test_model(model, test_loader, device, test_output_dir)

if __name__ == "__main__":
    main()
