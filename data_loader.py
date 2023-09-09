from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class ImageEnhancementDataset(Dataset):
    def __init__(self, input_root_dir, output_root_dir, transform=None):
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        self.transform = transform
        self.input_image_list = os.listdir(input_root_dir)
        self.output_image_list = os.listdir(output_root_dir)

    def __len__(self):
        return len(self.input_image_list)

    def __getitem__(self, idx):
        input_img_name = os.path.join(self.input_root_dir, self.input_image_list[idx])
        output_img_name = os.path.join(self.output_root_dir, self.output_image_list[idx])

        input_image = Image.open(input_img_name).convert('RGB')
        output_image = Image.open(output_img_name).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image

def get_data_loaders(train_input_root_dir, train_output_root_dir, test_input_root_dir, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = ImageEnhancementDataset(train_input_root_dir, train_output_root_dir, transform=transform)
    test_dataset = ImageEnhancementDataset(test_input_root_dir, test_input_root_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader