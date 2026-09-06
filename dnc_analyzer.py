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

Three epsilon values are returned:
  epsilon_1 : for NoiseInjector Stage 1 (pre-DC, raw input image)
  epsilon_2 : for NoiseInjector Stage 2 (pre-quantum, 128x4 features)
  epsilon_3 : for NoiseInjector Stage 3 (post-reconstruction, final 172x128x128 output)

Why is epsilon_3 smaller than epsilon_1 by default (stage3_scale=0.5)?
  - Stage 3 noise is added AFTER the image has already been fully
    reconstructed. Two noise injections (Stage 1, Stage 2) have already
    been composed into the output by this point. Adding a full-strength
    third dose on top would degrade visual/spectral quality more than
    it improves privacy, since the earlier stages already provided
    protection during the compute-heavy part of the pipeline.
  - The final-output injection is best understood as a smaller "sealing"
    layer -- enough to guarantee the released image also carries some
    protection, without stacking noise levels additively.

Formula used (Exponential Decay, the standard starting point in DP-SGD):
  epsilon_base = epsilon_max * exp(-decay_rate * progress)
  progress     = current_epoch / total_epochs  (a number from 0.0 to 1.0)
  epsilon_1    = epsilon_base
  epsilon_2    = epsilon_base * stage2_scale
  epsilon_3    = epsilon_base * stage3_scale
"""

import math

import torch
import torch.nn as nn


class DNCAnalyzer(nn.Module):
    """Decides the noise level (epsilon) for all three DNC injection points.

    Call `compute_epsilons(epoch, total_epochs)` at the START of every
    training step to get the epsilon values, then pass them into the
    three NoiseInjector modules via their `set_epsilon()` method.

    Parameters
    ----------
    epsilon_max : float
        Maximum noise level (applied at the very start of training,
        epoch 0). Default 0.15.
    decay_rate : float
        How fast the noise decays as training progresses. Default 3.0.
    stage2_scale : float
        Multiplier applied to get epsilon_2 from epsilon_base. Default 1.5.
    stage3_scale : float
        Multiplier applied to get epsilon_3 (post-reconstruction, final
        output) from epsilon_base. Default 0.5 -- final-output noise is
        deliberately lighter than Stage 1, since it's a "sealing" layer
        applied on top of an already-protected pipeline, not the primary
        protection mechanism. See module docstring for details.
    min_epsilon : float
        Floor value — epsilon will never go below this. Default 0.01.
    """

    def __init__(
        self,
        epsilon_max: float = 0.15,
        decay_rate: float = 3.0,
        stage2_scale: float = 1.5,
        stage3_scale: float = 0.5,
        min_epsilon: float = 0.01,
    ):
        super().__init__()

        if epsilon_max <= 0:
            raise ValueError(f"epsilon_max must be > 0, got {epsilon_max}")
        if decay_rate <= 0:
            raise ValueError(f"decay_rate must be > 0, got {decay_rate}")
        if stage2_scale <= 0:
            raise ValueError(f"stage2_scale must be > 0, got {stage2_scale}")
        if stage3_scale <= 0:
            raise ValueError(f"stage3_scale must be > 0, got {stage3_scale}")
        if min_epsilon < 0:
            raise ValueError(f"min_epsilon must be >= 0, got {min_epsilon}")

        self.epsilon_max = epsilon_max
        self.decay_rate = decay_rate
        self.stage2_scale = stage2_scale
        self.stage3_scale = stage3_scale
        self.min_epsilon = min_epsilon

        self._current_epsilon1 = epsilon_max
        self._current_epsilon2 = epsilon_max * stage2_scale
        self._current_epsilon3 = epsilon_max * stage3_scale
        self._discriminator_loss_weight = 0.0

    # ------------------------------------------------------------------
    # Core method — call this at the START of every training step
    # ------------------------------------------------------------------

    def compute_epsilons(
        self,
        current_epoch: int,
        total_epochs: int,
    ) -> tuple:
        """Compute the three epsilon values for this training step.

        Returns
        -------
        (epsilon_1, epsilon_2, epsilon_3) : (float, float, float)
            epsilon_1 -> NoiseInjector Stage 1 (pre-DC)
            epsilon_2 -> NoiseInjector Stage 2 (pre-quantum)
            epsilon_3 -> NoiseInjector Stage 3 (post-reconstruction, final output)
        """
        if total_epochs <= 0:
            raise ValueError(f"total_epochs must be > 0, got {total_epochs}")
        if current_epoch < 0:
            raise ValueError(
                f"current_epoch must be >= 0, got {current_epoch}"
            )

        progress = min(current_epoch / total_epochs, 1.0)

        epsilon_base = self.epsilon_max * math.exp(-self.decay_rate * progress)
        epsilon_base = epsilon_base * (1.0 + self._discriminator_loss_weight)
        epsilon_base = max(epsilon_base, self.min_epsilon)

        epsilon_1 = epsilon_base
        epsilon_2 = min(
            epsilon_base * self.stage2_scale,
            self.epsilon_max * self.stage2_scale
        )
        epsilon_2 = max(epsilon_2, self.min_epsilon)

        epsilon_3 = min(
            epsilon_base * self.stage3_scale,
            self.epsilon_max * self.stage3_scale
        )
        epsilon_3 = max(epsilon_3, self.min_epsilon)

        self._current_epsilon1 = epsilon_1
        self._current_epsilon2 = epsilon_2
        self._current_epsilon3 = epsilon_3

        return epsilon_1, epsilon_2, epsilon_3

    # ------------------------------------------------------------------
    # Optional upgrade — Discriminator loss integration
    # ------------------------------------------------------------------

    def update_from_discriminator_loss(self, d_loss: float):
        """Adjust epsilon based on the current Discriminator loss.
        (Unchanged from before — affects all three epsilons via epsilon_base.)
        """
        target_loss = 0.693  # ln(2), the balanced BCE point
        imbalance = (d_loss - target_loss) / target_loss
        self._discriminator_loss_weight = max(-0.2, min(0.2, imbalance * 0.2))

    # ------------------------------------------------------------------
    # Convenience / Logging
    # ------------------------------------------------------------------

    @property
    def current_epsilons(self) -> tuple:
        """Return the most recently computed (epsilon_1, epsilon_2, epsilon_3)."""
        return self._current_epsilon1, self._current_epsilon2, self._current_epsilon3

    def forward(self, x):
        raise NotImplementedError(
            "DNCAnalyzer is a controller, not a tensor transform. "
            "Call compute_epsilons(epoch, total_epochs) to get epsilon values, "
            "then pass those to NoiseInjector.set_epsilon()."
        )

    def extra_repr(self) -> str:
        return (
            f"epsilon_max={self.epsilon_max}, decay_rate={self.decay_rate}, "
            f"stage2_scale={self.stage2_scale}, stage3_scale={self.stage3_scale}, "
            f"min_epsilon={self.min_epsilon}"
        )


if __name__ == "__main__":
    analyzer = DNCAnalyzer()
    for epoch in [0, 25, 50, 75, 99]:
        e1, e2, e3 = analyzer.compute_epsilons(epoch, 100)
        print(f"epoch={epoch:>3}  e1={e1:.6f}  e2={e2:.6f}  e3={e3:.6f}")
