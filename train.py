import torch
import os
import re
from torchvision.utils import save_image
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, VisualInformationFidelity, PeakSignalNoiseRatio
from loss_functions import *

# Custom metric function for LNL1
def lnl1_metric(prediction, target):
    max = torch.maximum(prediction, target) + 1e-3 * torch.ones_like(prediction)
    norm1 = torch.absolute(prediction - target)
    normalize_norm1 = torch.mean(torch.mul(norm1, max.pow(-1)))
    lnl1_value = -20*torch.log10(normalize_norm1)

    return lnl1_value

# Extract the epoch number from a filename using regular expressions
def extract_epoch_number(filename):
    match = re.search(r"epoch(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0  # Default to 0 if no match is found

# Train a deep learning model
def train_model(model, train_loader, num_epochs, device, model_save_dir, generated_image_dir, criterion_str='PSNR',  save_every=5):
    # Choose the appropriate loss criterion based on the provided string
    if criterion_str == 'PSNR':
        criterion = PSNRLoss()
    elif criterion_str == 'SSIM':
        criterion = TangentSSIMLoss()
    elif criterion_str == 'LNL1':
        criterion = LNL1Loss()
    elif criterion_str == 'SSIM_PSNR_LNL1':
        criterion = PSNR_SSIM_LNL1Loss(1, 50, 1)
    else:
        raise ValueError("Unsupported loss criterion. Supported criteria are 'PSNR', 'SSIM', 'LNL1', and 'SSIM_PSNR_LNL1'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # Check if there are saved models in the model_save_dir
    checkpoint_files = [f for f in os.listdir(model_save_dir) if re.match(r'model_checkpoint_epoch\d+\.pth', f)]

    if checkpoint_files:
        epoch_numbers = [int(re.search(r'model_checkpoint_epoch(\d+)\.pth', f).group(1)) for f in checkpoint_files]
        epoch_numbers.sort()

        latest_epoch = epoch_numbers[-1]
        latest_checkpoint = f'model_checkpoint_epoch{latest_epoch:04d}.pth'
        checkpoint_path = os.path.join(model_save_dir, latest_checkpoint)

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['generator_state_dict'])  # Load the generator's weights
        optimizer.load_state_dict(checkpoint['optimizer_G_state_dict'])

        print(f"Loaded checkpoint from epoch {latest_epoch}. Resuming training...")
    else:
        latest_epoch = 0
        print("No checkpoint found. Starting training from epoch 1...")

        best_loss = float('inf')  # Initialize with negative infinity
        best_model_path = None

    # Initialize the VIF metric
    vif_metric = VisualInformationFidelity()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1)
    psnr_metric = PeakSignalNoiseRatio()
    best_loss = float('inf')

    model.train()

    for epoch in range(latest_epoch, num_epochs):
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        running_vif = 0.0
        running_lnl1 = 0.0

        for batch_idx, (input_batch, output_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)):
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

            # Report metrics after each 5 batches
            if (batch_idx + 1) % 5 == 0:
                average_loss = running_loss / (batch_idx + 1)
                average_psnr = running_psnr / (batch_idx + 1)
                average_ssim = running_ssim / (batch_idx + 1)
                average_vif = running_vif / (batch_idx + 1)
                average_lnl1 = running_lnl1 / (batch_idx + 1)
                save_image(torch.cat((outputs, output_batch), dim=0), f'{generated_image_dir}/images_epoch_{epoch + 1:03d}_batch_{batch_idx + 1:02d}.png')

                print(f"\nEpoch {epoch+1}/{num_epochs} Batch {batch_idx+1}/{len(train_loader)} - Loss: {average_loss:.4f} PSNR: {average_psnr:.4f} SSIM: {average_ssim:.4f} VIF: {average_vif:.4f} LNL1: {average_lnl1:.4f}")

        average_loss = running_loss / len(train_loader)
        average_psnr = running_psnr / len(train_loader)
        average_ssim = running_ssim / len(train_loader)
        average_vif = running_vif / len(train_loader)
        average_lnl1 = running_lnl1 / len(train_loader)

        # Save the model every 'save_every' epochs
        if (epoch + 1) % save_every == 0:
            save_path = f'{model_save_dir}/model_checkpoint_epoch{epoch + 1:04d}.pth'
            torch.save({
                'generator_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer.state_dict()
            }, save_path)
            print(f'Model is saved at epoch {epoch + 1}.')

        # Save the best model according to loss
        if average_loss < best_loss:
            best_loss = average_loss
            best_model_path = os.path.join(model_save_dir, f'best_cnn_image_enhancement_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'$$$ Best model is updated based on {criterion_str} $$$')
