"""
train_gpu.py
================================================================
HyperKING Federated-Privacy Project - GPU TRAINING SCRIPT
================================================================

Fixes in this version:
  1. Device mismatch: quantum modules (lightning.qubit) run on the CPU.
     Their outputs are now moved back to the GPU before the next layer
     (Generator: before Inverse-QC, Discriminator: before Sigmoid).
  2. Input to the Discriminator's quantum classifier is moved to CPU first.
  3. Uses the new torch.amp API (no more FutureWarnings).
  4. Plain ASCII text only (no weird characters on Windows).

USAGE
-----
    python train_gpu.py --data-dir .\\patches_400 --checkpoint-dir .\\checkpoints --log-dir .\\logs --epochs 2300 --batch-size 2 --checkpoint-every 25 --qdevice lightning.qubit

Resume after interruption:
    python train_gpu.py --data-dir .\\patches_400 --checkpoint-dir .\\checkpoints --log-dir .\\logs --epochs 2300 --batch-size 2 --checkpoint-every 25 --qdevice lightning.qubit --resume
"""

import os
import sys
import csv
import time
import signal
import argparse
import traceback
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------
# IMPORTS FROM YOUR REPO
# ----------------------------------------------------------------------
try:
    from patch_dataset import PatchDataset
    from generator import GeneratorFirstHalf
    from inverse_qc_module import InverseQCModule
    from lowrank_module import LowRankModule
    from noise_injector import NoiseInjector
    from dnc_analyzer import DNCAnalyzer
    from ds_module import DSModule
    from he_quantum_classifier import HEQuantumClassifier
    from sigmoid_module import SigmoidModule
    from losses import generator_loss, discriminator_loss
except ImportError as e:
    print("=" * 70)
    print("IMPORT ERROR - a module from your hyperking-src repo was not found.")
    print(f"  Missing: {e}")
    print("=" * 70)
    sys.exit(1)


def _device_of(module):
    """Return the device where a module's weights live."""
    return next(module.parameters()).device


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

class Generator(torch.nn.Module):
    def __init__(self, qdevice="lightning.qubit"):
        super().__init__()
        self.first_half = GeneratorFirstHalf(qdevice=qdevice)
        self.inverse_qc = InverseQCModule()
        self.low_rank = LowRankModule()
        self.stage3_dnc = DNCAnalyzer()
        self.stage3_noise = NoiseInjector()

    def step_dnc(self, current_epoch, total_epochs, discriminator_loss=None):
        self.first_half.step_dnc(current_epoch, total_epochs,
                                 discriminator_loss=discriminator_loss)
        if discriminator_loss is not None:
            self.stage3_dnc.update_from_discriminator_loss(discriminator_loss)
        _, _, e3 = self.stage3_dnc.compute_epsilons(current_epoch, total_epochs)
        self._epsilon_3 = e3

    def forward(self, x):
        # DC -> Reshape -> Core Quantum FE (+ DNC stage 1/2)
        x = self.first_half(x)
        # Quantum output may come back on CPU -> move to the GPU
        x = x.to(_device_of(self.inverse_qc))
        # Inverse-QC
        x = self.inverse_qc(x)
        # Low-rank -> 172x128x128
        x = self.low_rank(x)
        if self.training:
            eps3 = getattr(self, "_epsilon_3", None)
            if eps3 is not None:
                self.stage3_noise.set_epsilon(eps3)
            x = self.stage3_noise(x)  # Novelty 1, Stage 3
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, qdevice="lightning.qubit"):
        super().__init__()
        self.ds = DSModule()
        self.he_classifier = HEQuantumClassifier(qdevice=qdevice)
        self.sigmoid = SigmoidModule()

    def forward(self, x):
        gpu_device = _device_of(self.sigmoid)
        x = self.ds(x)
        # Quantum classifier runs on CPU -> send input to CPU
        x = self.he_classifier(x.cpu())
        # Bring the quantum output back to the GPU, as float32
        x = x.to(gpu_device).float()
        x = self.sigmoid(x)
        return x


# ----------------------------------------------------------------------
# Checkpointing helpers
# ----------------------------------------------------------------------

def save_checkpoint(path, epoch, generator, discriminator, opt_g, opt_d):
    torch.save({
        "epoch": epoch,
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "opt_g_state": opt_g.state_dict(),
        "opt_d_state": opt_d.state_dict(),
    }, path)


def find_latest_checkpoint(checkpoint_dir):
    ckpts = sorted(Path(checkpoint_dir).glob("epoch_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


_shutdown_requested = {"flag": False}


def _handle_sigterm(signum, frame):
    print("\n[signal] SIGTERM received - will checkpoint at end of current epoch and exit.")
    _shutdown_requested["flag"] = True


signal.signal(signal.SIGTERM, _handle_sigterm)


# ----------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HyperKING GAN training")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--epochs", type=int, default=2300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--alternate-period", type=int, default=5)
    parser.add_argument("--warmup-g-epochs", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--qdevice", default="lightning.qubit",
                        choices=["default.qubit", "lightning.qubit", "lightning.gpu"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ---------------- Device ----------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[device] WARNING: no CUDA GPU detected - training will run on CPU and be slow.")

    # ---------------- Data ----------------
    print(f"[data] Loading patches from {args.data_dir}")
    dataset = PatchDataset(patches_dir=args.data_dir)
    print(f"[data] {len(dataset)} patches found")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        drop_last=True)

    # ---------------- Models ----------------
    generator = Generator(qdevice=args.qdevice).to(device)
    discriminator = Discriminator(qdevice=args.qdevice).to(device)

    opt_g = optim.RMSprop(generator.parameters(), lr=args.lr)
    opt_d = optim.RMSprop(discriminator.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if latest is not None:
            print(f"[resume] Loading checkpoint {latest}")
            ckpt = torch.load(latest, map_location=device)
            generator.load_state_dict(ckpt["generator_state"])
            discriminator.load_state_dict(ckpt["discriminator_state"])
            opt_g.load_state_dict(ckpt["opt_g_state"])
            opt_d.load_state_dict(ckpt["opt_d_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"[resume] Resuming from epoch {start_epoch}")
        else:
            print("[resume] --resume was passed but no checkpoint found - starting fresh.")

    # ---------------- Logging ----------------
    log_path = os.path.join(args.log_dir, "loss_log.csv")
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["epoch", "g_loss", "d_loss", "epsilon_1",
                             "epsilon_2", "epsilon_3", "seconds"])

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"[train] Starting training: epochs {start_epoch} -> {args.epochs - 1}, "
          f"batch size {args.batch_size}, {len(loader)} batches/epoch")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        generator.train()
        discriminator.train()

        generator.step_dnc(current_epoch=epoch, total_epochs=args.epochs)

        train_d_this_epoch = (epoch >= args.warmup_g_epochs and
                              (epoch - args.warmup_g_epochs) % args.alternate_period == 0)

        running_g, running_d, n_batches = 0.0, 0.0, 0

        for corrupted, clean in loader:
            corrupted = corrupted.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            # ---- Train Generator ----
            opt_g.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                restored = generator(corrupted)
                d_pred_fake = discriminator(restored)
                g_loss = generator_loss(restored, clean, d_pred_fake)
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()

            # ---- Train Discriminator (only on alternation epochs) ----
            if train_d_this_epoch:
                opt_d.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    d_pred_real = discriminator(clean)
                    d_pred_fake_detached = discriminator(restored.detach())
                    d_loss = discriminator_loss(d_pred_real, d_pred_fake_detached)
                scaler.scale(d_loss).backward()
                scaler.step(opt_d)
                scaler.update()
                running_d += d_loss.item()

            running_g += g_loss.item()
            n_batches += 1

        avg_g = running_g / max(n_batches, 1)
        avg_d = (running_d / max(n_batches, 1)) if train_d_this_epoch else float("nan")
        elapsed = time.time() - epoch_start

        eps = generator.first_half.dnc if hasattr(generator.first_half, "dnc") else None
        eps1 = getattr(eps, "epsilon_1", None) if eps else None
        eps2 = getattr(eps, "epsilon_2", None) if eps else None
        eps3 = getattr(generator, "_epsilon_3", None)

        d_text = f"{avg_d:.5f}" if train_d_this_epoch else "skip"
        print(f"[epoch {epoch:4d}/{args.epochs}] g_loss={avg_g:.5f} "
              f"d_loss={d_text:>8} time={elapsed:.1f}s")

        log_writer.writerow([epoch, avg_g, avg_d, eps1, eps2, eps3, round(elapsed, 2)])
        log_file.flush()

        if ((epoch + 1) % args.checkpoint_every == 0
                or epoch == args.epochs - 1
                or _shutdown_requested["flag"]):
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, epoch, generator, discriminator, opt_g, opt_d)
            print(f"[checkpoint] Saved {ckpt_path}")

        if _shutdown_requested["flag"]:
            print("[shutdown] Exiting cleanly after checkpoint.")
            break

    log_file.close()
    print("[train] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("=" * 70)
        print("TRAINING CRASHED - full traceback below. Your last checkpoint is safe")
        print("in --checkpoint-dir; re-run with --resume to continue from there.")
        print("=" * 70)
        traceback.print_exc()
        sys.exit(1)
