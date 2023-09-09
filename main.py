import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# Define a custom dataset for loading high-resolution image pairs
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

# Define a CNN-based model for image enhancement
class CNNImageEnhancementModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNImageEnhancementModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        upscaled = self.upscale(encoded)
        decoded = self.decoder(upscaled)
        return decoded


# Load your training and test datasets
train_input_root_dir = 'data/DIV2K_train_HR_50x40'  # Low-resolution input
train_output_root_dir = 'data/DIV2K_train_HR_100x80'  # High-resolution output
test_input_root_dir = 'data/DIV2K_valid_HR_100x80'  # Test set high-resolution input

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = ImageEnhancementDataset(train_input_root_dir, train_output_root_dir, transform=transform)
test_dataset = ImageEnhancementDataset(test_input_root_dir, test_input_root_dir, transform=transform)

# DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Instantiate the model
model = CNNImageEnhancementModel(input_channels=3, output_channels=3)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
        input_batch, output_batch = batch
        input_batch, output_batch = input_batch.to(device), output_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_batch)

        loss = criterion(outputs, output_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    average_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {average_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'cnn_image_enhancement_model.pth')

# Testing phase
test_output_dir = 'result'

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
