import os
import torch
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# Define the test_model function
def test_model(model, test_loader, device, test_output_dir, model_checkpoint_path):
    # Load the best model checkpoint
    if os.path.exists(model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path)
        model.load_state_dict(checkpoint)
        print(f"Loaded best model checkpoint from {model_checkpoint_path}")
    else:
        print(f"No checkpoint found at {model_checkpoint_path}. Make sure to provide the correct path.")

    model.eval()
    if not os.path.exists(test_output_dir):
        os.mkdir(test_output_dir)

    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader)):  # Add tqdm here
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Convert tensors to NumPy arrays
            output_np = outputs.cpu().numpy().squeeze(0)
            target_np = targets.cpu().numpy().squeeze(0)

            # Calculate PSNR and SSIM for this sample
            psnr_value = psnr(target_np, output_np)
            ssim_value = ssim(target_np, output_np, channel_axis=0, data_range=1)
            
            psnr_scores.append(psnr_value)
            ssim_scores.append(ssim_value)

            # Save the enhanced image and optionally the real image for comparison
            filename = os.path.join(test_output_dir, f"enhanced_{i + 1:04d}.png")
            save_image(outputs.squeeze(0), filename)


    # Compute the mean PSNR and SSIM scores
    mean_psnr = np.mean(psnr_scores)
    mean_ssim = np.mean(ssim_scores)

    print(f"Mean PSNR between enhanced images and real ones: {mean_psnr:.4f}")
    print(f"Mean SSIM between enhanced images and real ones: {mean_ssim:.4f}")
