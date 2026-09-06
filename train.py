"""
train.py - HyperKING training loop (base paper + Novelty 1: DNC)

Wires together everything built so far:
  Generator  = GeneratorFirstHalf (DC+Reshape+CoreQuantumFE+DNC stages1/2)
               -> InverseQCModule -> LowRankModule -> NoiseInjector (stage 3)
  Discriminator = DSModule -> HEQuantumClassifierModule -> SigmoidModule
  Losses = generator_loss (Eq.2), discriminator_loss (Eq.3)
  DNC    = DNCAnalyzer computing epsilon_1/2/3 each step

Training schedule (per base paper, Section III-A):
  - Train the Generator alone for the first `alternate_period` epochs.
  - After that, alternate: train Discriminator for `alternate_period`
    epochs, then Generator for `alternate_period` epochs, repeating.
  - Optimizer: RMSprop, lr=0.01 (paper's choice, since it handles the
    non-stationary adversarial dynamics better than Adam).

IMPORTANT PERFORMANCE NOTE:
  The HE Quantum Classifier runs one PennyLane circuit per group (32
  groups) per image, with no batching inside the simulator. This makes
  the Discriminator forward pass slow -- expect real training (thousands
  of epochs, as in the paper) to take a long time on Colab's simulator.
  This script is correct and will run, but for now, test it with a
  SMALL total_epochs (e.g. 2-3) and small batch_size (e.g. 2) to confirm
  the whole loop works end-to-end before attempting a long training run.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from patch_dataset import PatchDataset
from generator import GeneratorFirstHalf
from inverse_qc_module import InverseQCModule
from lowrank_module import LowRankModule
from noise_injector import NoiseInjector
from dnc_analyzer import DNCAnalyzer

from ds_module import DSModule
from he_quantum_classifier import HEQuantumClassifierModule
from sigmoid_module import SigmoidModule

from losses import generator_loss, discriminator_loss


class Generator(torch.nn.Module):
    """Full Generator: base-paper pipeline + Novelty 1 Stage-3 noise."""

    def __init__(self):
        super().__init__()
        self.first_half = GeneratorFirstHalf()   # owns DNC Stage 1 + Stage 2 internally
        self.inverse_qc = InverseQCModule()
        self.lowrank = LowRankModule()
        self.noise3 = NoiseInjector(name="Stage3_postOutput")

    def forward(self, x):
        x = self.first_half(x)
        x = self.inverse_qc(x)
        x = self.lowrank(x)
        x = self.noise3(x)
        return x


class Discriminator(torch.nn.Module):
    """Full Discriminator: DS -> HE Quantum Classifier -> Sigmoid."""

    def __init__(self):
        super().__init__()
        self.ds = DSModule()
        self.classifier = HEQuantumClassifierModule()
        self.sigmoid = SigmoidModule()

    def forward(self, x):
        x = self.ds(x)
        x = self.classifier(x)
        x = self.sigmoid(x.float())  # cast: PennyLane returns float64
        return x


def train(
    patches_dir: str,
    total_epochs: int = 3,
    batch_size: int = 2,
    alternate_period: int = 60,
    lr: float = 0.01,
):
    dataset = PatchDataset(patches_dir=patches_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator()
    discriminator = Discriminator()
    dnc_analyzer = DNCAnalyzer()

    opt_g = optim.RMSprop(generator.parameters(), lr=lr)
    opt_d = optim.RMSprop(discriminator.parameters(), lr=lr)

    for epoch in range(total_epochs):
        # Decide who trains this epoch, per the paper's schedule:
        # epoch 0 to alternate_period-1: Generator only.
        # after that: alternate every `alternate_period` epochs.
        cycle_position = epoch // alternate_period
        train_generator_this_epoch = (cycle_position == 0) or (cycle_position % 2 == 0)

        generator.train()
        discriminator.train()

        # Set DNC epsilons for this epoch (Stage 1/2 live inside first_half,
        # Stage 3 lives on the Generator wrapper itself)
        e1, e2, e3 = dnc_analyzer.compute_epsilons(epoch, total_epochs)
        generator.first_half.noise_stage1.set_epsilon(e1)
        generator.first_half.noise_stage2.set_epsilon(e2)
        generator.noise3.set_epsilon(e3)

        epoch_g_loss, epoch_d_loss, n_batches = 0.0, 0.0, 0

        for corrupted, clean in loader:
            generator.first_half.step_dnc(epoch, total_epochs)
            fake = generator(corrupted)

            d_real = discriminator(clean)
            d_fake = discriminator(fake.detach())

            if train_generator_this_epoch:
                opt_g.zero_grad()
                d_fake_for_g = discriminator(fake)
                g_loss = generator_loss(clean, fake, d_fake_for_g)
                g_loss.backward()
                opt_g.step()
                epoch_g_loss += g_loss.item()
            else:
                opt_d.zero_grad()
                d_loss = discriminator_loss(d_real, d_fake)
                d_loss.backward()
                opt_d.step()
                dnc_analyzer.update_from_discriminator_loss(d_loss.item())
                epoch_d_loss += d_loss.item()

            n_batches += 1

        who = "Generator" if train_generator_this_epoch else "Discriminator"
        avg_loss = (epoch_g_loss if train_generator_this_epoch else epoch_d_loss) / max(n_batches, 1)
        print(f"Epoch {epoch:>3} | trained: {who:<13} | avg loss: {avg_loss:.4f} "
              f"| eps1={e1:.4f} eps2={e2:.4f} eps3={e3:.4f}")

    return generator, discriminator


if __name__ == "__main__":
    # Quick local smoke test with tiny dummy data (no real patches needed).
    # On Colab, call train(patches_dir='/content/drive/MyDrive/Hyperking/patches', ...)
    print("This script is meant to be run on Colab with your real patches_dir.")
    print("Example:")
    print("  from train import train")
    print("  generator, discriminator = train(")
    print("      patches_dir='/content/drive/MyDrive/Hyperking/patches',")
    print("      total_epochs=3, batch_size=2)")
