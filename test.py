import os
import torch
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure, VisualInformationFidelity, PeakSignalNoiseRatio
from tqdm import tqdm

# Define the test_model function
def test_model(model, test_loader, device, test_output_dir, model_checkpoint_path):
    # Load the best model checkpoint if available
    if os.path.exists(model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path)
        model.load_state_dict(checkpoint)
        print(f"Loaded the best model checkpoint from {model_checkpoint_path}")
    else:
        print(f"No checkpoint found at {model_checkpoint_path}. Please provide the correct path.")

    # Set the model to evaluation mode
    model.eval()

    # Create the test output directory if it doesn't exist
    if not os.path.exists(test_output_dir):
        os.mkdir(test_output_dir)


    # Set metrics
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarityIndexMeasure(data_range=1)
    vif = VisualInformationFidelity()

    psnr_scores = []
    ssim_scores = []
    vif_scores = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader)):  # Add tqdm progress bar
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)


            # Calculate PSNR and SSIM for this sample
            psnr_value = psnr(targets, outputs)
            ssim_value = ssim(targets, outputs)
            vif_value = vif(targets,outputs)

            psnr_scores.append(psnr_value)
            ssim_scores.append(ssim_value)
            vif_scores.append(vif_value)

            # Save the enhanced image
            filename = os.path.join(test_output_dir, f"enhanced_{i + 1:04d}.png")
            save_image(outputs.squeeze(0), filename)

    # Compute the mean PSNR and SSIM scores for the entire dataset
    mean_psnr = np.mean(psnr_scores)
    mean_ssim = np.mean(ssim_scores)
    mean_vif = np.mean(vif_scores)

    # Print the mean scores
    print(f"Mean PSNR between enhanced images and real ones: {mean_psnr:.4f}")
    print(f"Mean SSIM between enhanced images and real ones: {mean_ssim:.4f}")
    print(f"Mean VIF between enhanced images and real ones: {mean_vif:.4f}")
    
