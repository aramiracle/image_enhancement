from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import torch

class ImageEnhancementDataset(Dataset):
    def __init__(self, input_root_dir, output_root_dir, transform=None, train=True):
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        self.transform = transform
        self.input_image_list = os.listdir(input_root_dir)
        self.output_image_list = os.listdir(output_root_dir)
        self.train = train

    def __len__(self):
        return len(self.input_image_list)

    def add_gaussian_blur(self, image, kernel_size=5, sigma=0.5):
        if random.random() < 0.5:  # Apply blur with 50% probability
            image = transforms.GaussianBlur(kernel_size, sigma)(image)
        return image

    def add_gaussian_noise(self, image, mean=0, std=0.001):
        if random.random() < 0.5:  # Apply noise with 50% probability
            image = transforms.ToTensor()(image)
            noise = torch.randn_like(image) * std + mean
            image = image + noise
            image = transforms.ToPILImage()(image)  # Convert back to PIL Image
        return image

    def __getitem__(self, idx):
        input_img_name = os.path.join(self.input_root_dir, self.input_image_list[idx])
        output_img_name = os.path.join(self.output_root_dir, self.output_image_list[idx])

        input_image = Image.open(input_img_name).convert('RGB')
        output_image = Image.open(output_img_name).convert('RGB')

        if self.train:
            # Apply random rotation to both input and output images
            angle = random.randint(-45, 45)  # You can adjust the rotation range as needed
            input_image = input_image.rotate(angle)
            output_image = output_image.rotate(angle)

            input_image = self.add_gaussian_blur(input_image)
            input_image = self.add_gaussian_noise(input_image)

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image
