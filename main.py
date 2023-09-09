import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Define a custom dataset for loading image pairs (original and enhanced)
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

# Define a CNN-Transformer model for image enhancement with fixed input and output size
class CNNTransformerModel(nn.Module):
    def __init__(self, cnn_backbone, transformer_encoder, output_size):
        super(CNNTransformerModel, self).__init__()
        self.cnn_backbone = cnn_backbone
        self.transformer_encoder = transformer_encoder
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 3, kernel_size=3, padding=1)
        )
        self.output_size = output_size

    def forward(self, x):
        features = self.cnn_backbone(x)
        # Reshape the features to match the expected input size of the transformer
        features = features.view(features.size(0), -1, 2048).permute(1, 0, 2)
        encoded_features = self.transformer_encoder(features)
        # Reshape the encoded features back to the original size
        encoded_features = encoded_features.permute(1, 0, 2).view(features.size(1), -1, *self.output_size, 2048)
        enhanced_image = self.decoder(encoded_features.permute(0, 4, 1, 2, 3).reshape(-1, *self.output_size, 3))
        return enhanced_image

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust to your fixed size
    transforms.ToTensor(),
])

# Load your training dataset from the provided input and output directories
input_root_dir = 'data/DIV2K_train_HR_1_8'
output_root_dir = 'data/DIV2K_train_HR_1_4'
train_dataset = ImageEnhancementDataset(input_root_dir, output_root_dir, transform=transform)

# DataLoader for training
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Specify the fixed input and output size
input_size = (3, 256, 256)  # Channels x Height x Width
output_size = (3, 256, 256)  # Channels x Height x Width

# Instantiate the CNN backbone (e.g., ResNet-50)
cnn_backbone = models.regnet_y_16gf(pretrained=True)
cnn_backbone = nn.Sequential(*list(cnn_backbone.children())[:-2])  # Remove last two layers

# Instantiate the Transformer encoder (you can configure its layers as needed)
transformer_encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=2048, nhead=8), num_layers=4
)

# Instantiate the CNN-Transformer model with fixed input and output size
model = CNNTransformerModel(cnn_backbone, transformer_encoder, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch in train_dataloader:
        input_batch, output_batch = batch
        input_batch, output_batch = input_batch.to(device), output_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_batch)
        loss = criterion(outputs, output_batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'image_enhancement_model.pth')

# Testing phase
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust to your fixed size
    transforms.ToTensor(),
])

test_root_dir = 'data/DIV2K_valid_HR_1_8'
test_dataset = ImageEnhancementDataset(test_root_dir, output_root_dir, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()  # Set the model to evaluation mode
test_loss = 0.0

with torch.no_grad():
    for batch in test_dataloader:
        input_batch, output_batch = batch
        input_batch, output_batch = input_batch.to(device), output_batch.to(device)
        outputs = model(input_batch)
        loss = criterion(outputs, output_batch)
        test_loss += loss.item()

# Calculate the average test loss
average_test_loss = test_loss / len(test_dataloader)

print(f"Average Test Loss: {average_test_loss}")
