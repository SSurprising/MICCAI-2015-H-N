import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, groups, paddings=1, mid_channels=None, dimension=2):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        if groups > out_channels:
            groups = out_channels

        if dimension == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=paddings),
                nn.PReLU(mid_channels),

                nn.Conv2d(mid_channels, out_channels, 3, padding=paddings),
                nn.PReLU(out_channels),

                nn.GroupNorm(groups, out_channels)
            )

        elif dimension == 3:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, 3, padding=paddings),
                nn.PReLU(mid_channels),

                nn.Conv3d(mid_channels, out_channels, 3, padding=paddings),
                nn.PReLU(out_channels),

                nn.GroupNorm(groups, out_channels)
            )

    def forward(self, inputs):
        return self.conv(inputs)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=2, stride=2, dimension=2):
        super().__init__()

        if dimension == 2:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, stride),
                nn.PReLU(out_channels),
            )
        elif dimension == 3:
            self.down = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel, stride),
                nn.PReLU(out_channels),
            )

    def forward(self, inputs):
        return self.down(inputs)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=2, stride=2, dimension=2):
        super().__init__()

        if dimension == 2:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel, stride),
                nn.PReLU(out_channels),
            )
        elif dimension == 3:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel, stride),
                nn.PReLU(out_channels),
            )

    def forward(self, inputs):
        return self.up(inputs)