import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange


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
    
""""
class VisionTransformerEncoder(nn.Module):
    def __init__(self, input_channels, patch_size, embed_dim, num_heads, num_layers):
        super(VisionTransformerEncoder, self).__init__()

        self.patch_embedding = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Calculate the number of patches based on your input size and patch size
        num_patches = (input_size // patch_size) ** 2
        
        # Adjust the positional embedding size to match the number of patches
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.positional_embedding
        x = rearrange(x, 'b c h w -> b (h w) c')  # Flatten spatial dimensions
        x = self.transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=(input_size // patch_size), w=(input_size // patch_size))
        return x
# Vision Transformer Decoder
class VisionTransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(VisionTransformerDecoder, self).__init__()

        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, memory):
        for layer in self.transformer_layers:
            x = layer(x, memory)
        return x

# Complete Transformer-based Image Enhancement Model
class TransformerImageEnhancementModel(nn.Module):
    def __init__(self, input_channels, output_channels, patch_size=16, embed_dim=512, num_heads=8, num_layers=6):
        super(TransformerImageEnhancementModel, self).__init__()

        # Vision Transformer Encoder
        self.encoder = VisionTransformerEncoder(input_channels, patch_size, embed_dim, num_heads, num_layers)

        # Upscaling layer
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Convolutional layer after upscaling and before decoding
        self.conv_before_decode = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # Vision Transformer Decoder
        self.decoder = VisionTransformerDecoder(embed_dim, num_heads, num_layers)

        # Final convolutional layer for output
        self.final_conv = nn.Conv2d(embed_dim, output_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)

        # Upscaling
        upscaled = self.upscale(encoded.permute(0, 2, 1).unsqueeze(1)).squeeze(1).permute(0, 2, 1)

        # Convolution after upscaling
        decoded = self.conv_before_decode(upscaled)

        # Decoder
        decoded = self.decoder(decoded, encoded)

        # Final output
        final_output = self.final_conv(decoded)
        return self.sigmoid(final_output)
"""