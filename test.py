import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CNNImageEnhancementModel
from data_loader import ImageEnhancementDataset
import os

def test_model(model, test_loader, device, test_output_dir):
    model.eval()
    os.makedirs(test_output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (input_image, _) in enumerate(test_loader):
            input_image = input_image.to(device)
            enhanced_image = model(input_image)
            enhanced_image = enhanced_image.squeeze().cpu()
            
            output_filename = os.path.join(test_output_dir, f"enhanced_{i + 1:04d}.png")
            transforms.ToPILImage()(enhanced_image).save(output_filename)

    print("Testing complete. Enhanced images are saved in the 'result' folder.")
