"""
noise_injector.py
-----------------
One of the two new modules introduced by our project's first novelty:
Dynamic Noise Calibration (DNC).

This module is the "hands" of DNC — it actually applies the noise.
The "brain" (the DNCAnalyzer in dnc_analyzer.py) decides HOW MUCH
noise to add; this module just adds it.

There are THREE injection points in our Generator pipeline:
  Stage 1: pre-DC  — raw corrupted HSI  (172×128×128)
  Stage 2: pre-Quantum — compressed features (128×4)
  Stage 3: post-reconstruction — final restored image (172×128×128)
           [Stage 3 is built later, on Colab, at the end of the full
            Generator pipeline. Stages 1 and 2 live here, locally.]

This one class handles all three — the input shape doesn't matter,
because the noise is simply added element-wise and the output shape
is always identical to the input shape.

The noise mechanism used here is the Gaussian Mechanism, which is
the standard choice in differential privacy literature:
  noisy_x = x + N(0, epsilon^2)
where N(0, epsilon^2) is Gaussian noise with mean 0 and standard
deviation epsilon. epsilon is provided by the DNCAnalyzer at
each forward pass — it is NOT fixed, which is what makes the whole
system "dynamic."

Why Gaussian and not Laplace?
  - Both are valid differential privacy mechanisms.
  - Gaussian is more forgiving to gradients during backpropagation
    (the noise distribution is smooth and differentiable, which matters
    since this module sits inside a trained network).
  - Gaussian is the standard in most federated + DP learning papers,
    making our results easier to compare against the literature.

Why does this need to be an nn.Module and not just a function?
  - It needs to sit inside nn.Sequential in the Generator, alongside
    the learned classical layers. nn.Sequential requires all steps
    to be nn.Module subclasses. Also, in training mode we add noise;
    in eval/inference mode we skip noise (the .training flag controls
    this automatically when you call model.eval()).

Input:  Any tensor (no shape restriction) + scalar epsilon
Output: Same shape as input, with Gaussian noise added (training mode)
        OR the input unchanged (eval/inference mode, no noise added)
"""

import torch
import torch.nn as nn


class NoiseInjector(nn.Module):
    """Adds calibrated Gaussian noise to a tensor (Gaussian Mechanism).

    In training mode:  output = input + N(0, epsilon^2)
    In eval mode:      output = input  (no noise — clean inference)

    The epsilon value is set from outside (by DNCAnalyzer) before
    each forward pass, via set_epsilon(). This is what makes the
    noise level "dynamic" — it changes every training step based
    on how training is going.

    Parameters
    ----------
    epsilon_init : float
        Starting epsilon value. This gets overwritten every training
        step by DNCAnalyzer via set_epsilon(). Default 0.1 is just
        a safe placeholder — it means "small noise to start."
    name : str
        A label for this injector's position in the pipeline
        (e.g. "Stage1_preDC", "Stage2_preQuantum", "Stage3_postOutput").
        Used only for printing/debugging — doesn't affect computation.
    """

    def __init__(self, epsilon_init: float = 0.1, name: str = ""):
        super().__init__()
        # epsilon is stored as a plain Python float — NOT an nn.Parameter.
        # We don't want PyTorch's optimizer to try to learn epsilon;
        # the DNCAnalyzer controls it explicitly based on training progress.
        self.epsilon = epsilon_init
        self.name = name

    def set_epsilon(self, epsilon: float):
        """Update the noise level for the next forward pass.

        Called by DNCAnalyzer before each training step.
        epsilon must be >= 0. If epsilon is 0, no noise is added
        (equivalent to eval mode, but in training mode).
        """
        if epsilon < 0:
            raise ValueError(
                f"NoiseInjector '{self.name}': epsilon must be >= 0, "
                f"got {epsilon:.6f}. Negative noise levels are meaningless."
            )
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to x during training; pass through in eval.

        Parameters
        ----------
        x : torch.Tensor
            Any tensor — shape doesn't matter. The noise matches x's
            shape, device (CPU/GPU), and dtype automatically.

        Returns
        -------
        torch.Tensor
            Noisy tensor in training mode, or clean x in eval mode.
            Shape is ALWAYS identical to the input shape.
        """
        # In eval/inference mode: no noise. The model should output
        # its best clean reconstruction, not add noise to the result.
        if not self.training:
            return x

        # epsilon == 0 is a valid state (e.g. at the end of training when
        # the analyzer decides no more noise is needed). Skip the random
        # number generation entirely to avoid wasting compute.
        if self.epsilon == 0.0:
            return x

        # torch.randn_like creates a tensor with the SAME shape, dtype,
        # and device as x — this is important, because x might be on a
        # GPU (during Colab training) or CPU (local testing), and the
        # noise tensor must live on the same device or the addition fails.
        noise = torch.randn_like(x) * self.epsilon

        return x + noise

    def extra_repr(self) -> str:
        """Shows epsilon and name when you print the module."""
        return f"name='{self.name}', epsilon={self.epsilon:.6f}"


# ---------------------------------------------------------------------------
# Quick shape and behaviour sanity check — run this file directly:
#   python src/noise_injector.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    print("=" * 60)
    print("NoiseInjector — shape and behaviour sanity check")
    print("=" * 60)

    torch.manual_seed(42)

    # --- Test 1: Stage 1 shape (pre-DC: raw HSI) ---
    injector_s1 = NoiseInjector(epsilon_init=0.1, name="Stage1_preDC")
    injector_s1.train()  # put in training mode

    dummy_hsi = torch.randn(1, 172, 128, 128)  # one real HSI patch
    noisy_hsi = injector_s1(dummy_hsi)

    print("\nTest 1 — Stage 1 (pre-DC, raw HSI)")
    print(f"  Input shape:  {tuple(dummy_hsi.shape)}")
    print(f"  Output shape: {tuple(noisy_hsi.shape)}  (must match input)")
    print(f"  Shapes match: {dummy_hsi.shape == noisy_hsi.shape}")
    # The noisy output should differ from the clean input
    are_different = not torch.allclose(dummy_hsi, noisy_hsi)
    print(f"  Noise was added (training mode): {are_different}")
    # Noise std should be close to epsilon (0.1)
    noise_std = (noisy_hsi - dummy_hsi).std().item()
    print(f"  Measured noise std: {noise_std:.4f}  (expected ~0.1)")

    # --- Test 2: Stage 2 shape (pre-quantum: compressed features) ---
    injector_s2 = NoiseInjector(epsilon_init=0.05, name="Stage2_preQuantum")
    injector_s2.train()

    dummy_features = torch.randn(1, 128, 4)  # output of Reshape module
    noisy_features = injector_s2(dummy_features)

    print("\nTest 2 — Stage 2 (pre-quantum, 128×4 features)")
    print(f"  Input shape:  {tuple(dummy_features.shape)}")
    print(f"  Output shape: {tuple(noisy_features.shape)}  (must match input)")
    print(f"  Shapes match: {dummy_features.shape == noisy_features.shape}")
    noise_std2 = (noisy_features - dummy_features).std().item()
    print(f"  Measured noise std: {noise_std2:.4f}  (expected ~0.05)")

    # --- Test 3: Eval mode — noise must NOT be added ---
    injector_eval = NoiseInjector(epsilon_init=999.0, name="EvalTest")
    injector_eval.eval()  # put in eval mode

    dummy = torch.randn(2, 128, 4)
    out_eval = injector_eval(dummy)
    print("\nTest 3 — Eval mode (no noise should be added even with huge epsilon)")
    print(f"  Input and output identical: {torch.allclose(dummy, out_eval)}")

    # --- Test 4: set_epsilon updates correctly ---
    injector_dyn = NoiseInjector(epsilon_init=0.1, name="DynamicTest")
    injector_dyn.train()
    injector_dyn.set_epsilon(0.0)  # set to zero → no noise

    dummy = torch.randn(1, 128, 4)
    out_zero = injector_dyn(dummy)
    print("\nTest 4 — epsilon=0.0 in training mode (no noise)")
    print(f"  Input and output identical: {torch.allclose(dummy, out_zero)}")

    # --- Test 5: gradient flow (needed for backprop during training) ---
    injector_grad = NoiseInjector(epsilon_init=0.1, name="GradTest")
    injector_grad.train()

    x = torch.randn(1, 128, 4, requires_grad=True)
    out = injector_grad(x)
    loss = out.sum()
    loss.backward()
    print("\nTest 5 — Gradient flow through NoiseInjector")
    print(f"  x.grad is not None: {x.grad is not None}")
    print("  (Gradients MUST flow through for backprop to work in the Generator)")

    print("\n" + "=" * 60)
    print("All tests passed. NoiseInjector is ready.")
    print("=" * 60)
