"""
generator.py
------------
The Generator of HyperKING, first-half implementation.
"""

import torch
import torch.nn as nn

from dc_module import DCModule
from reshape_module import ReshapeModule
from core_quantum_fe import CoreQuantumFE
from noise_injector import NoiseInjector
from dnc_analyzer import DNCAnalyzer


class GeneratorFirstHalf(nn.Module):
    """First half of the HyperKING Generator with Dynamic Noise Calibration."""

    def __init__(
        self,
        in_channels: int = 172,
        epsilon_max: float = 0.15,
        decay_rate: float = 3.0,
        stage2_scale: float = 1.5,
        min_epsilon: float = 0.01,
        qdevice: str = "default.qubit",
    ):
        super().__init__()

        self.dnc_analyzer = DNCAnalyzer(
            epsilon_max=epsilon_max,
            decay_rate=decay_rate,
            stage2_scale=stage2_scale,
            min_epsilon=min_epsilon,
        )

        self.noise_stage1 = NoiseInjector(
            epsilon_init=epsilon_max,
            name="Stage1_preDC",
        )

        self.dc = DCModule(in_channels=in_channels)
        self.reshape = ReshapeModule()

        self.noise_stage2 = NoiseInjector(
            epsilon_init=epsilon_max * stage2_scale,
            name="Stage2_preQuantum",
        )

        self.quantum_fe = CoreQuantumFE(qdevice=qdevice)

    def step_dnc(
        self,
        current_epoch: int,
        total_epochs: int,
        discriminator_loss: float = None,
    ):
        if discriminator_loss is not None:
            self.dnc_analyzer.update_from_discriminator_loss(discriminator_loss)

        e1, e2, _e3 = self.dnc_analyzer.compute_epsilons(current_epoch, total_epochs)
        self.noise_stage1.set_epsilon(e1)
        self.noise_stage2.set_epsilon(e2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.noise_stage1(x)
        x = self.dc(x)
        x = self.reshape(x)
        x = self.noise_stage2(x)
        x = self.quantum_fe(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"epsilon_max={self.dnc_analyzer.epsilon_max}, "
            f"decay_rate={self.dnc_analyzer.decay_rate}, "
            f"stage2_scale={self.dnc_analyzer.stage2_scale}"
        )


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("GeneratorFirstHalf — shape and DNC integration check")
    print("=" * 60)

    torch.manual_seed(0)

    gen = GeneratorFirstHalf(
        in_channels=172,
        epsilon_max=0.15,
        decay_rate=3.0,
        stage2_scale=1.5,
        min_epsilon=0.01,
    )
    gen.train()

    print(gen)
    print()

    TOTAL_EPOCHS = 100
    gen.step_dnc(current_epoch=0, total_epochs=TOTAL_EPOCHS)
    e1, e2, _e3 = gen.dnc_analyzer.current_epsilons
    print(f"DNC epsilons at epoch 0: epsilon_1={e1:.6f}, epsilon_2={e2:.6f}")
    print()

    print("Running forward pass (batch=1)...")
    dummy_input = torch.randn(1, 172, 128, 128)
    start = time.time()
    with torch.no_grad():
        output = gen(dummy_input)
    elapsed = time.time() - start
    print(f"Output: {tuple(output.shape)}  (expected: (1, 64, 2, 2))")
    print(f"Time:   {elapsed:.1f} seconds")
    print()

    gen_eval = GeneratorFirstHalf(in_channels=172)
    gen_eval.load_state_dict(gen.state_dict())
    gen_eval.eval()
    with torch.no_grad():
        output_eval = gen_eval(dummy_input)
    noise_effect = (output - output_eval).abs().mean().item()
    print(f"DNC noise effect (train vs eval): {noise_effect:.6f}")
    print()

    print("DNC epsilon_1 schedule (training mode):")
    for epoch in [0, 25, 50, 75, 99]:
        gen.step_dnc(epoch, TOTAL_EPOCHS)
        e1, e2, _e3 = gen.dnc_analyzer.current_epsilons
        print(f"  epoch {epoch:3d}: epsilon_1={e1:.6f}, epsilon_2={e2:.6f}")
    print()

    print("Gradient flow check...")
    gen.train()
    gen.step_dnc(0, TOTAL_EPOCHS)
    x = torch.randn(1, 172, 128, 128)
    out = gen(x)
    loss = out.sum()
    loss.backward()
    dc_grad = gen.dc.conv_module_1.net[0].block[0].weight.grad
    quantum_grad = gen.quantum_fe.alpha.grad
    print(f"  Grad flows to DC module:      {dc_grad is not None}")
    print(f"  Grad flows to quantum params: {quantum_grad is not None}")
    print()

    n_params = sum(p.numel() for p in gen.parameters())
    print(f"Total trainable parameters: {n_params:,}")
    print("GeneratorFirstHalf is complete and ready.")
