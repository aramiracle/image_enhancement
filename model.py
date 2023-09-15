import torch
import torch.nn as nn
import torchvision.transforms as T

# Define a CNN-based model for image enhancement
class CNNImageEnhancementModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNImageEnhancementModel, self).__init__()
        # Encoder layers for feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Upsampling layer for image upscaling
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Decoder layers for image reconstruction
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward pass through the model
        encoded = self.encoder(x)
        upscaled = self.upscale(encoded)
        decoded = self.decoder(upscaled)
        return decoded

# Define a simpler CNN-based model for image enhancement
class SimpleCNNImageEnhancementModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleCNNImageEnhancementModel, self).__init__()
        # Encoder layers for feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Upsampling layer for image upscaling
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Decoder layers for image reconstruction
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward pass through the model
        encoded = self.encoder(x)
        upscaled = self.upscale(encoded)
        decoded = self.decoder(upscaled)
        return decoded

# Define a self-attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        attention_map = torch.matmul(query.permute(0, 2, 1), key)
        attention_map = torch.nn.functional.softmax(attention_map, dim=-1)

        out = torch.matmul(attention_map, value.permute(0, 2, 1)).view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

# Define a CNN-based model for image enhancement with self-attention
class AttentionCNNImageEnhancementModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(AttentionCNNImageEnhancementModel, self).__init__()
        # Encoder layers for feature extraction with self-attention
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(16),  # Apply self-attention here
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(32),  # Apply self-attention here
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(64)  # Apply self-attention here
        )
        # Upsampling layer for image upscaling
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Decoder layers for image reconstruction with self-attention
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(32),  # Apply self-attention here
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            SelfAttention(16),  # Apply self-attention here
            nn.Conv2d(16, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward pass through the model
        encoded = self.encoder(x)
        upscaled = self.upscale(encoded)
        decoded = self.decoder(upscaled)
        return decoded

class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample by 2x
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # Additional convolution
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)