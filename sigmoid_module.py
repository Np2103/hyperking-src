"""Sigmoid Module - Table IV, HyperKING base paper.
Two linear+sigmoid stages: 1x128 -> 1x16 -> 1x1 (real/fake probability)."""

import torch
import torch.nn as nn


class SigmoidModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Linear(128, 16), nn.Sigmoid())
        self.stage2 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        # x: (batch, 1, 128) -> squeeze to (batch, 128)
        x = x.squeeze(1)
        x = self.stage1(x)   # (batch, 16)
        x = self.stage2(x)   # (batch, 1)
        return x


if __name__ == "__main__":
    model = SigmoidModule()
    dummy = torch.randn(4, 1, 128)
    out = model(dummy)
    print("Output shape:", out.shape)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()
    print("Shape check passed, values in [0,1]")
