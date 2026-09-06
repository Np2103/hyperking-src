"""DS (Downsampling) Module - Table IV, HyperKING base paper.
BatchNorm -> LeakyReLU -> blended adaptive max+avg pooling (0.6/0.4).
172x128x128 -> 2x16x16"""

import torch
import torch.nn as nn


class DSModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(172)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.max_pool = nn.AdaptiveMaxPool2d((16, 16))
        self.avg_pool = nn.AdaptiveAvgPool2d((16, 16))
        # project 172 channels -> 2 channels (needed to hit the table's 2x16x16 output)
        self.channel_reduce = nn.Conv2d(172, 2, kernel_size=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.lrelu(x)
        x = self.channel_reduce(x)          # (batch, 2, 128, 128)
        pooled = 0.6 * self.max_pool(x) + 0.4 * self.avg_pool(x)  # (batch, 2, 16, 16)
        return pooled


if __name__ == "__main__":
    model = DSModule()
    dummy = torch.randn(4, 172, 128, 128)
    out = model(dummy)
    print("Output shape:", out.shape)
    assert out.shape == (4, 2, 16, 16)
    print("Shape check passed")
