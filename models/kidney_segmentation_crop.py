import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
import importlib.util


# === U-Net definition ===
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = block(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = block(64, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.dec2 = block(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.dec1 = block(64, 32)
        self.out = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat((self.up2(b), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.out(d1)
