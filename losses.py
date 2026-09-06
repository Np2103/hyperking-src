"""
Loss functions for HyperKING - Eq. (2) and (3) of the base paper.

Generator loss (Eq. 2):
    L_G = ||I_real - I_fake||_S + lambda * log(1 - D(I_fake) + delta)
    - ||.||_S is the smoothed L1 norm (reconstruction term)
    - second term rewards G for making D output HIGH real-probability on fakes
    - lambda = 0.01, delta = 1e-8 (paper's values)

Discriminator loss (Eq. 3):
    L_D = -[ log(D(I_real) + delta) + log(1 - D(I_fake) + delta) ]
    - standard entropy loss: rewards D for correctly labeling real as real
      and fake as fake
"""

import torch
import torch.nn as nn


def smoothed_l1_norm(x: torch.Tensor) -> torch.Tensor:
    """Smoothed L1 (Huber-style) norm, summed over all elements.
    Behaves like L2 for small differences (smooth near 0, good gradients)
    and like L1 for large differences (robust to outliers)."""
    abs_x = torch.abs(x)
    quadratic = 0.5 * x ** 2
    linear = abs_x - 0.5
    loss = torch.where(abs_x < 1.0, quadratic, linear)
    return loss.sum()


def generator_loss(I_real, I_fake, D_fake_prob, lam=0.01, delta=1e-8):
    """
    I_real, I_fake: (batch, 172, 128, 128) real and generated images
    D_fake_prob: (batch, 1) Discriminator's real-probability for I_fake
    """
    reconstruction_term = smoothed_l1_norm(I_real - I_fake)
    adversarial_term = torch.log(1 - D_fake_prob + delta).mean()
    return reconstruction_term + lam * adversarial_term


def discriminator_loss(D_real_prob, D_fake_prob, delta=1e-8):
    """
    D_real_prob: (batch, 1) Discriminator's real-probability for real images
    D_fake_prob: (batch, 1) Discriminator's real-probability for fake images
    """
    real_term = torch.log(D_real_prob + delta).mean()
    fake_term = torch.log(1 - D_fake_prob + delta).mean()
    return -(real_term + fake_term)


if __name__ == "__main__":
    torch.manual_seed(0)
    I_real = torch.randn(4, 172, 128, 128)
    I_fake = torch.randn(4, 172, 128, 128)
    D_real_prob = torch.rand(4, 1)
    D_fake_prob = torch.rand(4, 1)

    g_loss = generator_loss(I_real, I_fake, D_fake_prob)
    d_loss = discriminator_loss(D_real_prob, D_fake_prob)

    print("Generator loss:", g_loss.item())
    print("Discriminator loss:", d_loss.item())
    assert torch.isfinite(g_loss)
    assert torch.isfinite(d_loss)
    print("Both losses are finite scalars — check passed")
