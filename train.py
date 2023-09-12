import torch
import os
import re
from tqdm import tqdm
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim  # Import SSIM from scikit-image

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

def calculate_ssim(prediction, target):
    # Calculate SSIM
    ssim_value = ssim(target.cpu().numpy(), prediction.cpu().numpy(), data_range=1.0, channel_axis=1)
    return ssim_value

def extract_epoch_number(filename):
    # Use regular expressions to extract the epoch number from the filename
    match = re.search(r"epoch(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0  # Default to 0 if no match is found

def train_model(model, train_loader, num_epochs, device, model_save_dir, save_every=5):
    criterion_psnr = PSNRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Check if there are saved models in the model_save_dir
    latest_epoch = 0
    saved_models = [filename for filename in os.listdir(model_save_dir) if filename.endswith('.pth')]
    if saved_models:
        # Find the latest saved model using the custom function
        latest_model = max(saved_models, key=extract_epoch_number)
        latest_epoch = extract_epoch_number(latest_model)
        
        # Load the latest model's weights
        model.load_state_dict(torch.load(os.path.join(model_save_dir, latest_model)))
        print(f"Loaded model from epoch {latest_epoch}")

    best_ssim = -float('inf')  # Initialize with negative infinity
    best_model_path = None

    for epoch in range(latest_epoch, num_epochs):
        model.train()
        running_loss_psnr = 0.0
        running_ssim = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
            input_batch, output_batch = batch
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_batch)

            loss_psnr = criterion_psnr(outputs, output_batch)
            running_loss_psnr += loss_psnr.item()

            # Calculate SSIM for this batch and accumulate
            ssim_value = calculate_ssim(outputs.detach(), output_batch.detach())
            running_ssim += ssim_value
        
            loss_psnr.backward()
            optimizer.step()
        
        average_loss_psnr = running_loss_psnr / len(train_loader)
        average_ssim = running_ssim / len(train_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - PSNR Loss: {average_loss_psnr:.4f}, SSIM: {average_ssim:.4f}")

        # Save the model every 'save_every' epochs
        if (epoch + 1) % save_every == 0:
            model_save_path = os.path.join(model_save_dir, f'cnn_image_enhancement_model_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1} to {model_save_path}")

        # Check if SSIM is better than the current best
        if average_ssim > best_ssim:
            best_ssim = average_ssim
            best_model_path = os.path.join(model_save_dir, f'best_cnn_image_enhancement_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print('Best model is updated')
    
    # Save the final best model based on SSIM
    if best_model_path:
        print(f"Best model saved based on SSIM to {best_model_path}")
