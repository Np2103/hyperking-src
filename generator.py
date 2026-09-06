"""
generator.py
------------
The Generator of HyperKING, first-half implementation.

This file assembles the complete first half of the Generator pipeline —
from the raw corrupted hyperspectral image all the way through the Core
Quantum FE module — with our project's first novelty (Dynamic Noise
Calibration) inserted at its two pre-quantum injection points.

WHAT THIS FILE COVERS:
  The complete forward path from corrupted input to the quantum FE output.

                                        ← YOUR NOVELTY (DNC)
    Corrupted HSI  172×128×128
            │
    [DNC — Noise Injector Stage 1]      epsilon_1 from DNCAnalyzer
            │  172×128×128 (noisy)
    [DC Module]                         4 classical ConvModules
            │  128×2×2
    [Reshape Operator]                  pure tensor reshape, no weights
            │  128×4
    [DNC — Noise Injector Stage 2]      epsilon_2 from DNCAnalyzer
            │  128×4 (noisy)
    [Core Quantum FE]                   4-qubit PennyLane circuit
            │  64×2×2
            ↓
        STOP HERE  ← end of local scope
        (Inverse-QC, Low-rank, DNC Stage 3 continue on Colab)

WHAT COMES AFTER (on Colab):
  - Inverse-QC Module:  64×2×2  → 8×128×128  (transposed convolutions)
  - Low-Rank Module:    8×128×128 → 172×128×128  (spectral upsampling)
  - DNC Noise Injector Stage 3:  final output privacy protection

WHY THE SPLIT?
  Everything above the "STOP" line is fast — the DC, Reshape, and DNC
  modules are pure classical PyTorch and run in milliseconds even on CPU.
  The Core Quantum FE calls PennyLane 128× per image per batch — on a
  laptop CPU this takes minutes per step, which makes it impossible to
  train meaningfully. On Colab's GPU with the lightning.qubit backend
  this drops to seconds. The second half of the Generator (Inverse-QC,
  Low-rank) is also conv-heavy and benefits from GPU.

  For local development: use this file's __main__ block to confirm shapes
  are correct with a small dummy tensor, without running full training.

HOW DNC IS INTEGRATED:
  The DNCAnalyzer is NOT part of the nn.Sequential chain (it produces
  epsilon values, not tensors). The Generator owns one DNCAnalyzer and
  two NoiseInjectors. Before each training step, the training loop calls:
      generator.step_dnc(epoch, total_epochs)
  which asks the DNCAnalyzer for the current epsilon values and loads
  them into the two NoiseInjectors via set_epsilon(). The forward() pass
  then just runs normally — the Injectors already know their epsilon.

  For eval/inference: call model.eval() and the NoiseInjectors
  automatically pass the data through without adding any noise.
"""

import torch
import torch.nn as nn

from dc_module import DCModule
from reshape_module import ReshapeModule
from core_quantum_fe import CoreQuantumFE
from noise_injector import NoiseInjector
from dnc_analyzer import DNCAnalyzer


class GeneratorFirstHalf(nn.Module):
    """First half of the HyperKING Generator with Dynamic Noise Calibration.

    Covers the pipeline from raw corrupted HSI to Core Quantum FE output:
        Noise_S1 → DC → Reshape → Noise_S2 → CoreQuantumFE

    DNC noise levels are controlled by calling step_dnc() before each
    training step. In eval mode, noise is skipped automatically.

    Parameters
    ----------
    in_channels : int
        Number of spectral bands in the input HSI. Default 172 (AVIRIS).
    epsilon_max : float
        Peak noise level at the start of training. Passed to DNCAnalyzer.
        Default 0.15. Increase for stronger privacy; decrease if early
        training is unstable.
    decay_rate : float
        How quickly noise decays as training progresses. Default 3.0
        (noise at end of training ≈ 5% of peak). Passed to DNCAnalyzer.
    stage2_scale : float
        How much stronger Stage 2 noise is relative to Stage 1.
        Default 1.5 (Stage 2 = 1.5× Stage 1 at any given step).
    min_epsilon : float
        Minimum noise floor — epsilon never drops below this.
        Default 0.01 (ensures some minimum privacy guarantee always holds).
    """

    def __init__(
        self,
        in_channels: int = 172,
        epsilon_max: float = 0.15,
        decay_rate: float = 3.0,
        stage2_scale: float = 1.5,
        min_epsilon: float = 0.01,
    ):
        super().__init__()

        # ── DNC Controller (not a tensor processor — lives outside Sequential)
        self.dnc_analyzer = DNCAnalyzer(
            epsilon_max=epsilon_max,
            decay_rate=decay_rate,
            stage2_scale=stage2_scale,
            min_epsilon=min_epsilon,
        )

        # ── DNC Injection Point 1 — before DC module (raw HSI input)
        self.noise_stage1 = NoiseInjector(
            epsilon_init=epsilon_max,
            name="Stage1_preDC",
        )

        # ── Base paper modules (unchanged from HyperKING)
        self.dc = DCModule(in_channels=in_channels)
        self.reshape = ReshapeModule()

        # ── DNC Injection Point 2 — before quantum circuit (128×4 features)
        self.noise_stage2 = NoiseInjector(
            epsilon_init=epsilon_max * stage2_scale,
            name="Stage2_preQuantum",
        )

        # ── Core Quantum FE (the actual 4-qubit quantum circuit)
        self.quantum_fe = CoreQuantumFE()

    # ------------------------------------------------------------------
    # DNC step — call this BEFORE each training iteration
    # ------------------------------------------------------------------

    def step_dnc(
        self,
        current_epoch: int,
        total_epochs: int,
        discriminator_loss: float = None,
    ):
        """Update the noise levels for the upcoming training step.

        Call this at the START of each training iteration, BEFORE the
        forward pass. Example usage in your training loop:

            for epoch in range(total_epochs):
                for batch in dataloader:
                    generator.step_dnc(epoch, total_epochs)
                    # (after Discriminator has run):
                    # generator.step_dnc(epoch, total_epochs, d_loss=d_loss.item())
                    restored = generator(corrupted_batch)
                    ...

        Parameters
        ----------
        current_epoch : int
            Current epoch index (0-indexed).
        total_epochs : int
            Total number of training epochs.
        discriminator_loss : float, optional
            Most recent Discriminator loss value. If provided, the DNC
            Analyzer uses it to fine-tune epsilon (see DNCAnalyzer docs).
            If None (default), the decay formula alone is used.
        """
        if discriminator_loss is not None:
            self.dnc_analyzer.update_from_discriminator_loss(discriminator_loss)

        e1, e2 = self.dnc_analyzer.compute_epsilons(current_epoch, total_epochs)
        self.noise_stage1.set_epsilon(e1)
        self.noise_stage2.set_epsilon(e2)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the first-half Generator pipeline.

        In training mode: DNC noise is injected at both points.
        In eval mode:     DNC noise is skipped; pure forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Corrupted hyperspectral image.
            Shape: (B, 172, H, W) — typically (B, 172, 128, 128).

        Returns
        -------
        torch.Tensor
            Post-quantum feature map.
            Shape: (B, 64, 2, 2)
            (This feeds into the Inverse-QC module, built on Colab.)
        """
        # Stage 1: add noise to raw input BEFORE the DC module sees it
        x = self.noise_stage1(x)                  # (B, 172, 128, 128) — noisy

        # DC Module: classical compression
        x = self.dc(x)                             # (B, 128, 2, 2)

        # Reshape: reformat for quantum circuit input
        x = self.reshape(x)                        # (B, 128, 4)

        # Stage 2: add noise to compressed features BEFORE quantum circuit
        x = self.noise_stage2(x)                   # (B, 128, 4) — noisy

        # Core Quantum FE: 4-qubit circuit, 128 evaluations per image
        x = self.quantum_fe(x)                     # (B, 64, 2, 2)

        return x

    def extra_repr(self) -> str:
        return (
            f"epsilon_max={self.dnc_analyzer.epsilon_max}, "
            f"decay_rate={self.dnc_analyzer.decay_rate}, "
            f"stage2_scale={self.dnc_analyzer.stage2_scale}"
        )


# ---------------------------------------------------------------------------
# Shape and integration sanity check — run this file directly:
#   python src/generator.py
#
# NOTE: This only tests shapes with a tiny dummy input.
# The quantum circuit (CoreQuantumFE) runs 128 circuit evaluations per image,
# which is SLOW locally. This __main__ block uses 1 batch item to keep
# the wait manageable (expect ~30–90 seconds on a laptop CPU).
# Real training goes to Google Colab with the lightning.qubit backend.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("GeneratorFirstHalf — shape and DNC integration check")
    print("=" * 60)
    print()
    print("NOTE: The quantum circuit (CoreQuantumFE) inside this test")
    print("runs 128 circuit evaluations. On a laptop CPU this takes")
    print("~30–90 seconds. This is expected — real training uses Colab.")
    print()

    torch.manual_seed(0)

    gen = GeneratorFirstHalf(
        in_channels=172,
        epsilon_max=0.15,
        decay_rate=3.0,
        stage2_scale=1.5,
        min_epsilon=0.01,
    )
    gen.train()  # training mode — DNC noise will be added

    # Print the full module structure so we can visually verify everything
    # is wired in the right order
    print("Module structure:")
    print(gen)
    print()

    # Simulate epoch 0 DNC step (maximum noise, start of training)
    TOTAL_EPOCHS = 100
    gen.step_dnc(current_epoch=0, total_epochs=TOTAL_EPOCHS)
    e1, e2 = gen.dnc_analyzer.current_epsilons
    print(f"DNC epsilons at epoch 0: epsilon_1={e1:.6f}, epsilon_2={e2:.6f}")
    print(f"  (epsilon_1 should be ~{gen.dnc_analyzer.epsilon_max:.3f},")
    print(f"   epsilon_2 should be ~{gen.dnc_analyzer.epsilon_max * gen.dnc_analyzer.stage2_scale:.3f})")
    print()

    # Forward pass — SMALL batch (1 image) to keep runtime manageable
    print("Running forward pass (batch=1) — quantum circuit starting...")
    dummy_input = torch.randn(1, 172, 128, 128)
    print(f"Input:  {tuple(dummy_input.shape)}")
    start = time.time()
    with torch.no_grad():
        output = gen(dummy_input)
    elapsed = time.time() - start
    print(f"Output: {tuple(output.shape)}  (expected: (1, 64, 2, 2))")
    print(f"Time:   {elapsed:.1f} seconds")
    print()

    # Verify DNC is working: noisy output should differ from a no-noise run
    gen_eval = GeneratorFirstHalf(in_channels=172)
    gen_eval.load_state_dict(gen.state_dict())
    gen_eval.eval()  # eval mode → no noise
    with torch.no_grad():
        output_eval = gen_eval(dummy_input)
    noise_effect = (output - output_eval).abs().mean().item()
    print(f"DNC noise effect (mean absolute diff, train vs eval): {noise_effect:.6f}")
    print("  (Should be > 0 — confirms DNC is adding noise in training mode)")
    print()

    # DNC schedule check across epochs
    print("DNC epsilon_1 schedule (training mode):")
    for epoch in [0, 25, 50, 75, 99]:
        gen.step_dnc(epoch, TOTAL_EPOCHS)
        e1, e2 = gen.dnc_analyzer.current_epsilons
        print(f"  epoch {epoch:3d}: epsilon_1={e1:.6f}, epsilon_2={e2:.6f}")
    print()

    # Gradient flow check
    print("Gradient flow check (needed for training)...")
    gen.train()
    gen.step_dnc(0, TOTAL_EPOCHS)
    x = torch.randn(1, 172, 128, 128)
    out = gen(x)
    loss = out.sum()
    loss.backward()
    # Check that gradients flow back to the DC module's conv weights
    dc_grad = gen.dc.conv_module_1.net[0].block[0].weight.grad
    quantum_grad = gen.quantum_fe.alpha.grad
    print(f"  Grad flows to DC module:      {dc_grad is not None}")
    print(f"  Grad flows to quantum params: {quantum_grad is not None}")
    print()

    n_params = sum(p.numel() for p in gen.parameters())
    print(f"Total trainable parameters (first-half Generator): {n_params:,}")
    print()
    print("=" * 60)
    print("GeneratorFirstHalf is complete and ready.")
    print("Next step: build Inverse-QC, Low-rank, DNC Stage 3 on Colab.")
    print("=" * 60)
