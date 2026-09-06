"""
dnc_analyzer.py
---------------
The "brain" of our project's first novelty: Dynamic Noise Calibration.

While noise_injector.py is the "hands" (it adds noise), this module is
the "brain" — it decides HOW MUCH noise to add at each injection point,
at each training step.

The key word in "Dynamic Noise Calibration" is DYNAMIC: the noise level
is NOT fixed upfront. It changes continuously during training, based on
two signals:
  1. How far along training is (epoch / total_epochs)
  2. How well the Discriminator is currently performing (its loss)

Why does dynamic noise matter?
  - At the START of training, the model needs strong privacy protection
    (high epsilon), because the raw, uncorrupted patterns in the training
    images haven't yet been blurred into the model weights. A strong
    attacker who can observe early gradients can still recover image
    content.
  - As training PROGRESSES, the model converges and the useful signal
    in any single training step becomes weaker. Continuing to add high
    noise at this point only hurts restoration quality without adding
    meaningful extra privacy. So we dial the noise DOWN.
  - The Discriminator loss is a direct signal of training quality:
    if the Discriminator is very confident (low loss → real vs fake is
    easy to tell apart), the Generator is still struggling, so we need
    to make sure we're not adding so much noise that it can't learn at all.

Two epsilon values are returned:
  epsilon_1 : for NoiseInjector Stage 1 (pre-DC, raw input image)
  epsilon_2 : for NoiseInjector Stage 2 (pre-quantum, 128×4 features)

Why two different values?
  - Stage 1 noise is added to the raw 172×128×128 image — a very large
    tensor with direct pixel values. A smaller epsilon here can already
    distort the image significantly (more numbers, each slightly noisy).
  - Stage 2 noise is added to the 128×4 compressed feature vectors —
    a much smaller tensor with abstracted features. These features are
    already far from raw pixel space, so they need a slightly higher
    epsilon to carry enough noise at the quantum level.
  - The ratio between epsilon_1 and epsilon_2 is controlled by the
    `stage2_scale` parameter (default: 1.5 — Stage 2 gets 50% more noise
    than Stage 1 at any given training point).

Formula used (Exponential Decay, the standard starting point in DP-SGD):
  epsilon_base = epsilon_max * exp(-decay_rate * progress)
  progress     = current_epoch / total_epochs  (a number from 0.0 to 1.0)
  epsilon_1    = epsilon_base
  epsilon_2    = epsilon_base * stage2_scale

  At progress=0.0 (start):  epsilon_base = epsilon_max  (maximum noise)
  At progress=1.0 (end):    epsilon_base = epsilon_max * exp(-decay_rate)
  With decay_rate=3.0:      final noise  ≈ 5% of the starting noise

This can be upgraded to a small MLP that also reads the Discriminator
loss directly — the architecture is designed for that upgrade (see
`update_from_discriminator_loss()` below) — but the formula is the right
starting point because:
  a) It's mathematically well-motivated (known from DP-SGD literature).
  b) It's explainable in a project report without needing to justify
     training a second network inside the training loop.
  c) It will work correctly even before you have a Discriminator loss
     to read from (i.e. right now, when you're testing the Generator alone).
"""

import math

import torch
import torch.nn as nn


class DNCAnalyzer(nn.Module):
    """Decides the noise level (epsilon) for both DNC injection points.

    Call `compute_epsilons(epoch, total_epochs)` at the START of every
    training step to get the epsilon values, then pass them into the
    two NoiseInjector modules via their `set_epsilon()` method.

    Parameters
    ----------
    epsilon_max : float
        Maximum noise level (applied at the very start of training,
        epoch 0). Default 0.15 — tuned so that early-training noise is
        visible but doesn't completely destroy the signal going into the
        DC module. Adjust upward for stronger privacy, downward if
        early training becomes unstable.
    decay_rate : float
        How fast the noise decays as training progresses. Higher value =
        faster decay. At decay_rate=3.0 (default), noise at epoch 100%
        is about 5% of the starting noise. At decay_rate=1.0, it would
        be about 37%.
    stage2_scale : float
        Multiplier applied to get epsilon_2 (pre-quantum Stage 2) from
        epsilon_base. Default 1.5 means Stage 2 gets 50% more noise than
        Stage 1 at any given training point. See module docstring for why.
    min_epsilon : float
        Floor value — epsilon will never go below this, even at the very
        end of training. This ensures some minimum privacy guarantee is
        always maintained. Default 0.01.
    """

    def __init__(
        self,
        epsilon_max: float = 0.15,
        decay_rate: float = 3.0,
        stage2_scale: float = 1.5,
        min_epsilon: float = 0.01,
    ):
        super().__init__()

        if epsilon_max <= 0:
            raise ValueError(f"epsilon_max must be > 0, got {epsilon_max}")
        if decay_rate <= 0:
            raise ValueError(f"decay_rate must be > 0, got {decay_rate}")
        if stage2_scale <= 0:
            raise ValueError(f"stage2_scale must be > 0, got {stage2_scale}")
        if min_epsilon < 0:
            raise ValueError(f"min_epsilon must be >= 0, got {min_epsilon}")

        self.epsilon_max = epsilon_max
        self.decay_rate = decay_rate
        self.stage2_scale = stage2_scale
        self.min_epsilon = min_epsilon

        # Internal state — updated on each call to compute_epsilons()
        # and optionally adjusted by the Discriminator loss signal.
        self._current_epsilon1 = epsilon_max
        self._current_epsilon2 = epsilon_max * stage2_scale
        self._discriminator_loss_weight = 0.0  # starts unused; see below

    # ------------------------------------------------------------------
    # Core method — call this at the START of every training step
    # ------------------------------------------------------------------

    def compute_epsilons(
        self,
        current_epoch: int,
        total_epochs: int,
    ) -> tuple:
        """Compute the two epsilon values for this training step.

        Parameters
        ----------
        current_epoch : int
            The current training epoch (0-indexed, so first epoch = 0).
        total_epochs : int
            Total number of training epochs planned.

        Returns
        -------
        (epsilon_1, epsilon_2) : (float, float)
            epsilon_1 → pass to NoiseInjector Stage 1 (pre-DC)
            epsilon_2 → pass to NoiseInjector Stage 2 (pre-quantum)
        """
        if total_epochs <= 0:
            raise ValueError(f"total_epochs must be > 0, got {total_epochs}")
        if current_epoch < 0:
            raise ValueError(
                f"current_epoch must be >= 0, got {current_epoch}"
            )

        # progress goes from 0.0 (start) to ~1.0 (end)
        progress = min(current_epoch / total_epochs, 1.0)

        # Exponential decay formula
        epsilon_base = self.epsilon_max * math.exp(-self.decay_rate * progress)

        # Apply the Discriminator-loss adjustment if it's been set
        # (see update_from_discriminator_loss() below).
        epsilon_base = epsilon_base * (1.0 + self._discriminator_loss_weight)

        # Apply the floor
        epsilon_base = max(epsilon_base, self.min_epsilon)

        epsilon_1 = epsilon_base
        epsilon_2 = min(
            epsilon_base * self.stage2_scale,
            self.epsilon_max * self.stage2_scale  # never exceed starting max
        )
        epsilon_2 = max(epsilon_2, self.min_epsilon)

        # Save for inspection / logging
        self._current_epsilon1 = epsilon_1
        self._current_epsilon2 = epsilon_2

        return epsilon_1, epsilon_2

    # ------------------------------------------------------------------
    # Optional upgrade — Discriminator loss integration
    # ------------------------------------------------------------------

    def update_from_discriminator_loss(self, d_loss: float):
        """Adjust epsilon based on the current Discriminator loss.

        This is the "upgrade path" described in the module docstring —
        once you have a working Discriminator and training loop (on Colab),
        you can call this BEFORE compute_epsilons() each step to make the
        noise adapt to how well the GAN is training, not just to the epoch.

        How the adjustment works:
          - High Discriminator loss → Discriminator is confused → Generator
            is doing well → we can AFFORD more noise (stronger privacy).
            Adjustment: slightly INCREASE epsilon.
          - Low Discriminator loss  → Discriminator is very confident →
            Generator is struggling → we should REDUCE noise so the
            Generator can still learn something useful.
            Adjustment: slightly DECREASE epsilon.

        The adjustment is deliberately SMALL (capped at ±20%) so it
        doesn't dominate over the base decay schedule.

        Parameters
        ----------
        d_loss : float
            The current Discriminator loss value (a positive float).
            Typical range depends on your loss function:
              - Binary cross-entropy: usually between 0.0 and ~2.0
              - Wasserstein distance: can be larger
            You'll know the typical range once training starts.
        """
        # Normalise the loss into a gentle ±0.2 adjustment weight.
        # This formula is deliberately simple — refine the scaling
        # factor (currently /2.0) once you see real d_loss values.
        # A well-balanced GAN has d_loss ≈ 0.693 (ln(2)) for BCE loss.
        # Below that = discriminator winning; above = generator winning.
        target_loss = 0.693  # ln(2), the balanced BCE point
        imbalance = (d_loss - target_loss) / target_loss  # -1 to +inf
        # Clamp to ±0.2 so the adjustment is at most a 20% change
        self._discriminator_loss_weight = max(-0.2, min(0.2, imbalance * 0.2))

    # ------------------------------------------------------------------
    # Convenience / Logging
    # ------------------------------------------------------------------

    @property
    def current_epsilons(self) -> tuple:
        """Return the most recently computed (epsilon_1, epsilon_2)."""
        return self._current_epsilon1, self._current_epsilon2

    def forward(self, x):
        """Not used directly — DNCAnalyzer is a controller, not a transform.

        It doesn't process tensors itself; it computes epsilon values and
        passes them to the NoiseInjectors, which do the actual tensor math.
        Raising NotImplementedError here is intentional — if you accidentally
        pass a tensor to this module, you'll get a clear error message.
        """
        raise NotImplementedError(
            "DNCAnalyzer is a controller, not a tensor transform. "
            "Call compute_epsilons(epoch, total_epochs) to get epsilon values, "
            "then pass those to NoiseInjector.set_epsilon()."
        )

    def extra_repr(self) -> str:
        return (
            f"epsilon_max={self.epsilon_max}, decay_rate={self.decay_rate}, "
            f"stage2_scale={self.stage2_scale}, min_epsilon={self.min_epsilon}"
        )


# ---------------------------------------------------------------------------
# Sanity check — run this file directly:
#   python src/dnc_analyzer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("DNCAnalyzer — epsilon schedule sanity check")
    print("=" * 60)

    analyzer = DNCAnalyzer(
        epsilon_max=0.15,
        decay_rate=3.0,
        stage2_scale=1.5,
        min_epsilon=0.01,
    )

    print("\nEpsilon schedule over 100 epochs:")
    print(f"  {'Epoch':>6}  {'Progress':>9}  {'epsilon_1':>10}  {'epsilon_2':>10}")
    print("  " + "-" * 44)

    TOTAL_EPOCHS = 100
    for epoch in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]:
        e1, e2 = analyzer.compute_epsilons(epoch, TOTAL_EPOCHS)
        progress = epoch / TOTAL_EPOCHS
        print(f"  {epoch:>6}  {progress:>9.2f}  {e1:>10.6f}  {e2:>10.6f}")

    # Confirm epsilon_1 at epoch 0 equals epsilon_max
    e1_start, _ = analyzer.compute_epsilons(0, TOTAL_EPOCHS)
    assert abs(e1_start - 0.15) < 1e-9, "epoch-0 epsilon_1 must equal epsilon_max"

    # Confirm epsilon_1 at last epoch is >= min_epsilon
    e1_end, _ = analyzer.compute_epsilons(99, TOTAL_EPOCHS)
    assert e1_end >= 0.01, "epsilon must never go below min_epsilon"

    # Confirm epsilon_2 > epsilon_1 always (Stage 2 gets more noise)
    for ep in range(0, 100, 10):
        e1, e2 = analyzer.compute_epsilons(ep, TOTAL_EPOCHS)
        assert e2 >= e1, f"epsilon_2 should be >= epsilon_1 at epoch {ep}"

    print("\nAll assertions passed.")

    # Show Discriminator-loss adjustment
    print("\nDiscriminator-loss adjustment demo:")
    analyzer2 = DNCAnalyzer()
    epoch = 30
    e1_base, e2_base = analyzer2.compute_epsilons(epoch, TOTAL_EPOCHS)
    print(f"  Base (no d_loss): e1={e1_base:.6f}, e2={e2_base:.6f}")

    analyzer2.update_from_discriminator_loss(1.5)  # discriminator winning
    e1_high, e2_high = analyzer2.compute_epsilons(epoch, TOTAL_EPOCHS)
    print(f"  After high d_loss (1.5 — gen winning):  e1={e1_high:.6f}, e2={e2_high:.6f}")

    analyzer2.update_from_discriminator_loss(0.2)  # generator struggling
    e1_low, e2_low = analyzer2.compute_epsilons(epoch, TOTAL_EPOCHS)
    print(f"  After low d_loss  (0.2 — disc winning): e1={e1_low:.6f}, e2={e2_low:.6f}")

    print("\n" + "=" * 60)
    print("DNCAnalyzer is ready.")
    print("=" * 60)
