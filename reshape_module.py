"""
reshape_module.py
------------------
Module 2 of the HyperKING Generator: the Reshape operator.

This is NOT a learned layer - no weights, no training. It's a pure
reformatting step that sits between the DC module's output and the
quantum circuit's input.

Why it exists: the quantum circuit encodes exactly 4 classical values
at a time (one per qubit, using 4 qubits via angle/amplitude
embedding). The DC module hands over a 128x2x2 feature map - 128
"channels," each a 2x2 spatial patch. This module just flattens each
2x2 patch into a length-4 vector, so you end up with 128 vectors of
length 4 instead of 128 grids of shape 2x2. Same 512 numbers, same
values, just reshaped so each qubit-ready group of 4 is contiguous.

Input:  (B, 128, 2, 2)
Output: (B, 128, 4)
"""

import torch
import torch.nn as nn


class ReshapeModule(nn.Module):
    """Flattens each 2x2 spatial patch into a length-4 vector.

    (B, 128, 2, 2) -> (B, 128, 4)

    No parameters - this is bookkeeping only. Implemented as an
    nn.Module (rather than a bare function) purely so it can sit
    inside an nn.Sequential alongside the DC module and later modules
    without special-casing it in the forward pass.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        assert (h, w) == (2, 2), (
            f"ReshapeModule expects a 2x2 spatial size (the DC module's "
            f"output), got {h}x{w}. Check what's feeding into this module."
        )
        # (B, 128, 2, 2) -> (B, 128, 4). Using reshape (not view) since
        # the DC module's output may not always be contiguous in memory.
        return x.reshape(b, c, h * w)


if __name__ == "__main__":
    # Shape sanity check, and a check that reshape doesn't scramble values -
    # each of the 4 numbers in a 2x2 patch should show up unchanged,
    # just flattened, in the same order every time (row-major: [0,0],
    # [0,1], [1,0], [1,1]).
    reshape = ReshapeModule()

    dummy = torch.randn(2, 128, 2, 2)
    out = reshape(dummy)
    print("Input shape: ", tuple(dummy.shape), "(expected: (2, 128, 2, 2))")
    print("Output shape:", tuple(out.shape), "(expected: (2, 128, 4))")

    # value-correctness check on one channel of one batch item
    original_patch = dummy[0, 0]  # shape (2, 2)
    flattened = out[0, 0]  # shape (4,)
    expected = torch.tensor(
        [original_patch[0, 0], original_patch[0, 1],
         original_patch[1, 0], original_patch[1, 1]]
    )
    matches = torch.allclose(flattened, expected)
    print(f"\nValues preserved correctly (row-major flatten): {matches}")

    # confirm chaining with the DC module works end-to-end
    try:
        from dc_module import DCModule

        dc = DCModule(in_channels=172)
        raw = torch.randn(2, 172, 128, 128)
        compressed = dc(raw)
        reshaped = reshape(compressed)
        print(f"\nEnd-to-end DC -> Reshape: {tuple(raw.shape)} -> "
              f"{tuple(compressed.shape)} -> {tuple(reshaped.shape)}")
        print("(expected: (2, 172, 128, 128) -> (2, 128, 2, 2) -> (2, 128, 4))")
    except ImportError:
        print(
            "\n(Skipping end-to-end DC->Reshape check - dc_module.py not "
            "found in the same folder. That's fine if you're just testing "
            "this file on its own; put both files in src\\ together to "
            "see the full chain.)"
        )