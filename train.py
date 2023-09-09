import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNNImageEnhancementModel
from data_loader import get_data_loaders
from tqdm import tqdm  # Import tqdm for the progress bar
import os
import torch
import torch.nn as nn

class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, prediction, target):
        # Calculate MSE (Mean Squared Error)
        mse = torch.mean((prediction - target) ** 2)
        
        # Calculate PSNR
        psnr = -10 * torch.log10(mse)
        
        # PSNR is typically used as a quality metric, so you want to minimize the negative PSNR
        # Invert the sign to use it as a loss
        return -psnr



def train_model(model, train_loader, num_epochs, device, model_save_dir):
    criterion = PSNRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Wrap train_loader with tqdm for the progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
            input_batch, output_batch = batch
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_batch)

            loss = criterion(outputs, output_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        average_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} Average Loss: {average_loss:.4f}")

    # Save the trained model
    model_save_path = os.path.join(model_save_dir, 'cnn_image_enhancement_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")
