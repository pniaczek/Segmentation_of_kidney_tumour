import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, features=[32, 64, 128]):
        super(UNet3D, self).__init__()
        self.encoder1 = self._block(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self._block(features[0], features[1])
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = self._block(features[1], features[2])

        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = self._block(features[2], features[1])
        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = self._block(features[1], features[0])

        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.encoder1(x)            
        e2 = self.encoder2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))

        u2 = self.up2(b)
        u2 = self.match_tensor_shape(u2, e2.shape)
        d2 = self.decoder2(torch.cat((u2, e2), dim=1))

        u1 = self.up1(d2)
        u1 = self.match_tensor_shape(u1, e1.shape)
        d1 = self.decoder1(torch.cat((u1, e1), dim=1))


        return self.out_conv(d1)

    
    @staticmethod
    def match_tensor_shape(tensor, ref_shape):
        _, _, d, h, w = tensor.shape
        td, th, tw = ref_shape[-3:]

        pad_d = max(0, td - d)
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)

        padding = [
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2
        ]
        tensor = F.pad(tensor, padding)

        _, _, d, h, w = tensor.shape
        start_d = (d - td) // 2
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        return tensor[:, :, start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]
