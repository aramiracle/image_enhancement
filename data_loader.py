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

    def add_gaussian_blur(self, image, kernel_size=5, sigma=1):
        if random.random() < 0.7:  # Apply blur with 70% probability
            image = transforms.GaussianBlur(kernel_size, sigma)(image)
        return image

    def add_gaussian_noise(self, image, mean=0, std=0.001):
        if random.random() < 0.7:  # Apply noise with 70% probability
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

        if self.transform:
            if self.train:  # Apply data augmentation only during training
                input_image = self.add_gaussian_blur(input_image)
                input_image = self.add_gaussian_noise(input_image)

            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image

def get_data_loaders(train_input_root_dir, train_output_root_dir, test_input_root_dir, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = ImageEnhancementDataset(train_input_root_dir, train_output_root_dir, transform=transform, train=True)
    test_dataset = ImageEnhancementDataset(test_input_root_dir, test_input_root_dir, transform=transform, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader
