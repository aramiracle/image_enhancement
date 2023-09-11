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
        self.train = train

        # List all input and output image file paths
        self.input_image_paths = [os.path.join(input_root_dir, fname) for fname in os.listdir(input_root_dir)]
        self.output_image_paths = [os.path.join(output_root_dir, fname) for fname in os.listdir(output_root_dir)]

    def __len__(self):
        return len(self.input_image_paths)

    def apply_data_augmentation(self, input_image, output_image):
        if self.train:
            # Random rotation
            angle = random.randint(-45, 45)
            input_image = input_image.rotate(angle)
            output_image = output_image.rotate(angle)

            # Color Jitter
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            input_image = color_jitter(input_image)
            output_image = color_jitter(output_image)

            # Gaussian blur and noise
            if random.random() < 0.5:
                input_image = transforms.GaussianBlur(5, 0.5)(input_image)
            if random.random() < 0.5:
                input_image = transforms.ToTensor()(input_image)
                noise = torch.randn_like(input_image) * 0.001 + 0
                input_image = input_image + noise
                input_image = transforms.ToPILImage()(input_image)

        return input_image, output_image

    def __getitem__(self, idx):
        input_img_path = self.input_image_paths[idx]
        output_img_path = self.output_image_paths[idx]

        input_image = Image.open(input_img_path).convert('RGB')
        output_image = Image.open(output_img_path).convert('RGB')

        input_image, output_image = self.apply_data_augmentation(input_image, output_image)

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image

def get_data_loaders(train_input_root_dir, train_output_root_dir, test_input_root_dir, test_output_root_dir, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = ImageEnhancementDataset(train_input_root_dir, train_output_root_dir, transform=transform, train=True)
    test_dataset = ImageEnhancementDataset(test_input_root_dir, test_output_root_dir, transform=transform, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader