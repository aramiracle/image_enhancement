import torch.nn as nn

# Define a Residual Block for the generator model.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # Define convolution layers and batch normalization.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Add the residual connection
        return out

# Define the Generator model for image enhancement.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            self.conv_block(3, 16),
            ResidualBlock(16, 16),  # Residual block in the encoder
            self.conv_block(16, 32),
            ResidualBlock(32, 32),
            self.conv_block(32, 64)
        )

        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)

        # Decoder with 2x resolution upscaling
        self.decoder = nn.Sequential(
            self.conv_block(64, 32),
            ResidualBlock(32, 32),  # Residual block in the decoder
            self.conv_block(32, 16),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)  # Final convolution to produce the output
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU activation
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU activation
        )
    
    def forward(self, x):
        x_enc = self.encoder(x)
        x_upsample = self.upsample(x_enc)
        output = self.decoder(x_upsample)
        return output

# Define a simpler Generator model for image enhancement.
class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            self.conv_block(3, 8),
            self.conv_block(8, 16),
            self.conv_block(16, 32),
        )

        self.upsample = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)

        # Decoder with 2x resolution upscaling
        self.decoder = nn.Sequential(
            self.conv_block(32, 16),
            self.conv_block(16, 8),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),  # Final convolution to produce the output
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU activation
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU activation
        )
    
    def forward(self, x):
        x_enc = self.encoder(x)
        x_upsample = self.upsample(x_enc)
        output = self.decoder(x_upsample)
        return output
