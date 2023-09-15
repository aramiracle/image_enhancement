# Train a deep learning model
import torch
import os
import re
from torchvision.utils import save_image
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, VisualInformationFidelity, PeakSignalNoiseRatio
from loss_functions import *


def lnl1_metric(prediction, target):
    max = torch.maximum(prediction, target) + 1e-3 * torch.ones_like(prediction)
    norm1 = torch.absolute(prediction - target)
    normalize_norm1 = torch.mean(torch.mul(norm1, max.pow(-1)))
    lnl1_value = -10*torch.log10(normalize_norm1)

    return lnl1_value

# Extract the epoch number from a filename using regular expressions
def extract_epoch_number(filename):
    match = re.search(r"epoch(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0  # Default to 0 if no match is found

# Train a deep learning model
def train_model(model, train_loader, num_epochs, device, model_save_dir, criterion_str='PSNR', save_every=5):
    if criterion_str == 'PSNR':
        criterion = PSNRLoss()
    elif criterion_str == 'SSIM':
        criterion = TangentSSIMLoss()
    elif criterion_str == 'LNL1':
        criterion = LNL1Loss()
    elif criterion_str == 'SSIM_PSNR_LNL1':
        criterion = PSNR_SSIM_LNL1Loss(1, 20, 2)
    else:
        raise ValueError("Unsupported loss criterion. Supported criteria are 'PSNR','SSIM', 'SSIM_PSNR' and 'SSIM_PSNR_NL1'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

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

    best_loss = float('inf')  # Initialize with negative infinity
    best_model_path = None

    # Initialize the VIF metric
    vif_metric = VisualInformationFidelity()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1)
    psnr_metric = PeakSignalNoiseRatio()

    for epoch in range(latest_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        running_vif = 0.0
        running_lnl1 = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
            input_batch, output_batch = batch
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_batch)

            loss = criterion(outputs, output_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Calculate PSNR for this batch and accumulate
            psnr = psnr_metric(outputs, output_batch)
            running_psnr += psnr
        
            # Calculate SSIM for this batch and accumulate
            
            ssim_value = ssim_metric(outputs, output_batch)
            running_ssim += ssim_value
            
            # Calculate VIF for this batch and accumulate
            vif_value = vif_metric(outputs, output_batch)
            running_vif += vif_value

            # Calculate LNL1 for this batch and accumulate
            lnl1_value = lnl1_metric(outputs, output_batch)
            running_lnl1 += lnl1_value

        average_loss = running_loss / len(train_loader)
        average_psnr = running_psnr / len(train_loader)
        average_ssim = running_ssim / len(train_loader)
        average_vif = running_vif / len(train_loader)
        average_lnl1 = running_lnl1 / len(train_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {average_loss:.4f}, PSNR: {average_psnr:.4f}, SSIM: {average_ssim:.4f}, VIF: {average_vif:.4f} LNL1: {average_lnl1:.4f}")

        # Save the model every 'save_every' epochs
        if (epoch + 1) % save_every == 0:
            model_save_path = os.path.join(model_save_dir, f'cnn_image_enhancement_model_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1} to {model_save_path}")


        # Save the best model according to loss
        if average_loss < best_loss:
            best_loss = average_loss
            best_model_path = os.path.join(model_save_dir, f'best_cnn_image_enhancement_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model is updated based on {criterion_str}')
            save_image(outputs, f'results/generated_images/epoch{epoch + 1:04d}.png')
    
    # Save the final best model based on the chosen criterion
    if best_model_path:
        print(f"Best model saved based on {criterion_str} to {best_model_path}")
