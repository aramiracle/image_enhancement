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


class TransformerImageEnhancementModel(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim, num_layers, num_heads, dropout=0.1, image_size=50):
        super(TransformerImageEnhancementModel, self).__init__()
        
        # Calculate positional encoding
        self.positional_encoding = self.calculate_positional_encoding(input_channels, image_size, hidden_dim)
        
        # Transformer Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, input_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        
        # Transformer Decoder Layers
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        
        # Final Linear Layer
        self.final_layer = nn.Linear(hidden_dim, output_channels)
        
    def calculate_positional_encoding(self, input_channels, image_size, hidden_dim):
        position = torch.arange(0, image_size * image_size).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        position_encoding = torch.zeros(1, image_size * image_size, hidden_dim)
        position_encoding[:, :, 0::2] = torch.sin(position * div_term)
        position_encoding[:, :, 1::2] = torch.cos(position * div_term)
        position_encoding = position_encoding.view(1, input_channels, image_size, image_size, hidden_dim)
        return position_encoding
        
    def forward(self, x):
        # Add positional encoding
        x = x + self.positional_encoding.to(x.device)
        
        # Transformer Encoder
        x_encoded = self.transformer_encoder(x)
        
        # Upsampling
        x_upsampled = self.upsample(x_encoded)
        
        # Transformer Decoder
        enhanced_image = self.transformer_decoder(x_upsampled)
        
        # Final Linear Layer
        enhanced_image = self.final_layer(enhanced_image)
        
        return enhanced_image