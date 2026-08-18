import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class LightweightFragmentUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_filters=16):
        """
        in_channels = 2 (CT Image + Click Point)
        out_channels = 2 (Core Mask + Boundary Mask for dual-head loss)
        """
        super().__init__()

        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base_filters, base_filters*2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base_filters*2, base_filters*4))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base_filters*4, base_filters*8))

        self.up1 = nn.ConvTranspose3d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_filters*8, base_filters*4)
        
        self.up2 = nn.ConvTranspose3d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_filters*4, base_filters*2)
        
        self.up3 = nn.ConvTranspose3d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_filters*2, base_filters)
        
        # Output Head
        self.outc = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        y = self.up1(x4)
        y = torch.cat([x3, y], dim=1)
        y = self.conv_up1(y)
        
        y = self.up2(y)
        y = torch.cat([x2, y], dim=1)
        y = self.conv_up2(y)
        
        y = self.up3(y)
        y = torch.cat([x1, y], dim=1)
        y = self.conv_up3(y)
        
        logits = self.outc(y)
        return logits
