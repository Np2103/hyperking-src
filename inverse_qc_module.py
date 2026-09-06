"""
Inverse-QC Module
==================
Base paper: HyperKING (Lin & Young, IEEE TGRS 2025), Table III.

Undoes the Quantum Collapse (QC) effect: takes the collapsed, measured
output of the Core Quantum FE Module and progressively reconstructs a
full-resolution (but still spectrally-compressed) feature map, mirroring
the DC module's downsampling path in reverse.

Architecture (from Table III / Module Architecture Reference):
    TConvModule 1: TCB(64,3)x1  + up-sample  -> 64x20x20
    TConvModule 2: TCB(64,3)x2, TCB(32,3)x2 + up-sample -> 32x56x56
    TConvModule 3: TCB(32,3)x2, TCB(16,3)x1 + up-sample -> 16x124x124
    TConvModule 4: TCB(16,3)x1, TCB(8,3)x1              -> 8x128x128

TCB(c, s) = TransposedConv2d(out_channels=c, kernel=s) -> BatchNorm2d -> LeakyReLU(0.2)

Note: the paper's Table III does not specify exact stride/padding for the
"bilinear up-sample" steps (only that bilinear interpolation is used and
gives the stated output sizes). To guarantee an exact match to the
documented shapes at every stage (rather than approximate them with a
fixed 2x scale factor, which does not land on 20/56/124/128 exactly from
a 2x2 input), each up-sample here targets the exact next-stage spatial
size via F.interpolate(..., size=...). All TCB conv layers use
kernel=3, stride=1, padding=1 (shape-preserving), so channel-depth
changes happen without perturbing the spatial size set by the
interpolation step.

Input:  (batch, 64, 2, 2)   -- collapsed output of Core Quantum FE Module
Output: (batch, 8, 128, 128) -- fed into the Low-rank Module (Er)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCB(nn.Module):
    """Transposed Conv Block: Conv2d (shape-preserving) -> BatchNorm2d -> LeakyReLU(0.2).

    Implemented with a stride-1, padding-1, kernel-3 conv (equivalent in
    effect to a shape-preserving transposed conv here since all spatial
    resizing is handled explicitly by the up-sample steps between blocks).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class InverseQCModule(nn.Module):
    def __init__(self):
        super().__init__()

        # TConvModule 1: 64x2x2 -> 64x20x20
        self.tconv1 = nn.Sequential(
            TCB(64, 64),
        )

        # TConvModule 2: 64x20x20 -> 32x56x56
        self.tconv2 = nn.Sequential(
            TCB(64, 64),
            TCB(64, 64),
            TCB(64, 32),
            TCB(32, 32),
        )

        # TConvModule 3: 32x56x56 -> 16x124x124
        self.tconv3 = nn.Sequential(
            TCB(32, 32),
            TCB(32, 32),
            TCB(32, 16),
        )

        # TConvModule 4: 16x124x124 -> 8x128x128 (no up-sample stage)
        self.tconv4 = nn.Sequential(
            TCB(16, 16),
            TCB(16, 8),
        )

    @staticmethod
    def _upsample_to(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # x: (batch, 64, 2, 2)
        x = self.tconv1(x)                      # (batch, 64, 2, 2)
        x = self._upsample_to(x, (20, 20))       # (batch, 64, 20, 20)

        x = self.tconv2(x)                       # (batch, 32, 20, 20)
        x = self._upsample_to(x, (56, 56))       # (batch, 32, 56, 56)

        x = self.tconv3(x)                       # (batch, 16, 56, 56)
        x = self._upsample_to(x, (124, 124))     # (batch, 16, 124, 124)

        x = self.tconv4(x)                       # (batch, 8, 128, 128) -- kernel3/pad1 preserves 124
        # NOTE: TConvModule4 has no explicit up-sample per Table III, but
        # input is 124x124 and output must be 128x128. We upsample once
        # more here to land on the documented final size exactly.
        x = self._upsample_to(x, (128, 128))
        return x


if __name__ == "__main__":
    model = InverseQCModule()
    dummy = torch.randn(4, 64, 2, 2)
    out = model(dummy)
    print("Input shape: ", dummy.shape)
    print("Output shape:", out.shape)
    assert out.shape == (4, 8, 128, 128), f"Shape mismatch: {out.shape}"
    print("Shape check passed: (4, 8, 128, 128)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
