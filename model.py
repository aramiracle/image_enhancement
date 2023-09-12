import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange
import math


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
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
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
    
    # Define a CNN-based model for image enhancement
class SimpleCNNImageEnhancementModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleCNNImageEnhancementModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        upscaled = self.upscale(encoded)
        decoded = self.decoder(upscaled)
        return decoded


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

    
class SimpleAttentionCNNImageEnhancementModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleAttentionCNNImageEnhancementModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(8),  # Add attention here
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(16),  # Add attention here
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(16),  # Add attention here
        )
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(8),  # Add attention here
            nn.Conv2d(8, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.upscale(x)
        x = self.decoder(x)
        return x