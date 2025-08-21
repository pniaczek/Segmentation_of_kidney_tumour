import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        self.downsample = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

        self.dropout = nn.Dropout3d(p=0.2) if dropout else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.dropout(out)

        out += identity
        return self.relu(out)


class Residual3DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, features=(32, 64, 128, 256)):
        super().__init__()

        self.encoder1 = ResidualBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(2)

        self.encoder2 = ResidualBlock(features[0], features[1])
        self.pool2 = nn.MaxPool3d(2)

        self.encoder3 = ResidualBlock(features[1], features[2])
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ResidualBlock(features[2], features[3])

        self.up3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(features[3], features[2])

        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(features[2], features[1])

        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(features[1], features[0])

        self.out = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.out(dec1)
