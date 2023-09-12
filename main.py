import torch
from train import train_model
from test import test_model
from sample_plt import display_random_samples
from data_loader import get_data_loaders
from model import *
import os

def main():
    model_name = 'attention'
    train_input_root_dir = 'data/DIV2K_train_HR/resized_25'
    train_output_root_dir = 'data/DIV2K_train_HR/resized_50'
    test_input_root_dir = 'data/DIV2K_valid_HR/resized_16'
    test_output_root_dir = 'data/DIV2K_valid_HR/resized_8'

    best_model_path = 'saved_models/'+ model_name +'/best_' + model_name + '_image_enhancement_model.pth'

    batch_size = 100
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_save_dir = 'saved_models/' + model_name
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    test_output_dir = 'results/' + model_name
    if not os.path.exists(test_output_dir):
        os.mkdir(test_output_dir)

    train_loader, test_loader = get_data_loaders(train_input_root_dir, train_output_root_dir, test_input_root_dir,test_output_root_dir, batch_size)

    model = AttentionCNNImageEnhancementModel(input_channels=3, output_channels=3).to(device)

    train_model(model, train_loader, num_epochs, device, model_save_dir)

    test_model(model, test_loader, device, test_output_dir, best_model_path)

    display_random_samples(test_output_dir, test_output_root_dir, num_samples=10)

if __name__ == "__main__":
    main()
