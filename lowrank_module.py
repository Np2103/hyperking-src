"""Low-rank Module (Er) - Table III, HyperKING base paper.
1x1 conv expanding 8x128x128 -> 172x128x128 (final restored HSI)."""

import torch
import torch.nn as nn


class LowRankModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.cb = nn.Sequential(
            nn.Conv2d(8, 172, kernel_size=1, padding=0),
            nn.BatchNorm2d(172),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv = nn.Conv2d(172, 172, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.cb(x)
        x = self.conv(x)
        return x


if __name__ == "__main__":
    model = LowRankModule()
    dummy = torch.randn(4, 8, 128, 128)
    out = model(dummy)
    print("Output shape:", out.shape)
    assert out.shape == (4, 172, 128, 128)
    print("Shape check passed")
