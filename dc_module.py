"""
dc_module.py
------------
Module 1 of the HyperKING Generator: the DC (Deep Compression) module.

Purpose: a 4-qubit NISQ quantum circuit can only accept 4 numbers at a
time. This module's entire job is to compress a 172x128x128 corrupted
hyperspectral cube (2,818,048 numbers) down to a 128x2x2 feature map
(512 numbers) small enough to feed the quantum circuit, using stacked
classical convolutions instead of PCA or naive patchwise splitting
(which the paper argues destroy spatial/spectral structure).

Input:  (B, 172, 128, 128)
Output: (B, 128, 2, 2)

Building block, per the paper's Table III:
    CB(c, s, p) = Conv2d(out_channels=c, kernel=s x s, padding=p)
                  -> BatchNorm2d -> LeakyReLU(negative_slope=0.2)
    MCB(c, k)   = 2x2 max-pool (stride 2) followed by a CB stack.

ConvModule 1 and ConvModule 2 are implemented exactly per the paper's
layer-by-layer spec (verified below to produce the paper's stated
16x124x124 and 32x56x56 intermediate shapes).

ConvModule 3 and ConvModule 4 follow the same maxpool+CB pattern
(channels doubling: 32 -> 64 -> 128), but the paper's text only gives
their FINAL output shapes (64x20x20, 128x2x2), not every intermediate
kernel/padding choice the way it does for Modules 1-2. Rather than
guess an exact stride/padding combination and risk silently landing on
the wrong spatial size, this implementation locks each module to its
documented target shape with a final AdaptiveAvgPool2d. This is a
clearly-flagged design choice, not a value copied from the paper -
if you get access to the exact Table III entries for ConvModule 3/4,
swap the AdaptiveAvgPool2d for the literal conv stack instead.
"""

import torch
import torch.nn as nn


class CB(nn.Module):
    """Conv2d -> BatchNorm2d -> LeakyReLU(0.2), per the paper's notation."""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvModule1(nn.Module):
    """172x128x128 -> 16x124x124, per the paper's exact spec."""

    def __init__(self, in_channels=172):
        super().__init__()
        self.net = nn.Sequential(
            CB(in_channels, 516, kernel_size=3, padding=1),  # 516x128x128
            CB(516, 172, kernel_size=3, padding=1),           # 172x128x128
            CB(172, 32, kernel_size=3, padding=1),             # 32x128x128
            CB(32, 16, kernel_size=3, padding=0),               # 16x126x126
            CB(16, 16, kernel_size=3, padding=0),               # 16x124x124
        )

    def forward(self, x):
        return self.net(x)


class ConvModule2(nn.Module):
    """16x124x124 -> 32x56x56, per the paper's exact spec."""

    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 124 -> 62
        self.net = nn.Sequential(
            CB(16, 16, kernel_size=3, padding=0),   # 62 -> 60
            CB(16, 32, kernel_size=3, padding=1),   # stays 60
            CB(32, 32, kernel_size=3, padding=0),   # 60 -> 58
            CB(32, 32, kernel_size=3, padding=0),   # 58 -> 56
        )

    def forward(self, x):
        x = self.maxpool(x)
        return self.net(x)


class ConvModule3(nn.Module):
    """32x56x56 -> 64x20x20 (target shape locked via AdaptiveAvgPool2d
    - see module docstring for why)."""

    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.net = nn.Sequential(
            CB(32, 32, kernel_size=3, padding=0),
            CB(32, 64, kernel_size=3, padding=1),
            CB(64, 64, kernel_size=3, padding=0),
            CB(64, 64, kernel_size=3, padding=0),
        )
        self.resize = nn.AdaptiveAvgPool2d((20, 20))

    def forward(self, x):
        x = self.maxpool(x)
        x = self.net(x)
        return self.resize(x)


class ConvModule4(nn.Module):
    """64x20x20 -> 128x2x2 (target shape locked via AdaptiveAvgPool2d
    - see module docstring for why)."""

    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.net = nn.Sequential(
            CB(64, 64, kernel_size=3, padding=0),
            CB(64, 128, kernel_size=3, padding=1),
            CB(128, 128, kernel_size=3, padding=0),
            CB(128, 128, kernel_size=3, padding=0),
        )
        self.resize = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        x = self.maxpool(x)
        x = self.net(x)
        return self.resize(x)


class DCModule(nn.Module):
    """Full Deep Compression module: 172x128x128 -> 128x2x2."""

    def __init__(self, in_channels=172):
        super().__init__()
        self.conv_module_1 = ConvModule1(in_channels)
        self.conv_module_2 = ConvModule2()
        self.conv_module_3 = ConvModule3()
        self.conv_module_4 = ConvModule4()

    def forward(self, x):
        x = self.conv_module_1(x)
        x = self.conv_module_2(x)
        x = self.conv_module_3(x)
        x = self.conv_module_4(x)
        return x


if __name__ == "__main__":
    # Quick shape sanity check - run this file directly to confirm the
    # module produces the paper's stated shapes at every stage.
    dc = DCModule(in_channels=172)
    dummy = torch.randn(2, 172, 128, 128)  # batch of 2, for good measure

    x = dummy
    x = dc.conv_module_1(x)
    print("After ConvModule1:", tuple(x.shape), "(expected: (2, 16, 124, 124))")
    x = dc.conv_module_2(x)
    print("After ConvModule2:", tuple(x.shape), "(expected: (2, 32, 56, 56))")
    x = dc.conv_module_3(x)
    print("After ConvModule3:", tuple(x.shape), "(expected: (2, 64, 20, 20))")
    x = dc.conv_module_4(x)
    print("After ConvModule4:", tuple(x.shape), "(expected: (2, 128, 2, 2))")

    out = dc(dummy)
    print("\nFull DCModule output:", tuple(out.shape), "(expected: (2, 128, 2, 2))")

    n_params = sum(p.numel() for p in dc.parameters())
    print(f"Total parameters: {n_params:,}")