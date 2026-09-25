"""
train_gpu.py
================================================================
HyperKING Federated-Privacy Project — GPU SERVER TRAINING SCRIPT
================================================================

This replaces your Colab notebook cells with a single script meant to be
run directly on a Linux GPU server via SSH (no notebook, no Drive mount).

WHAT THIS SCRIPT ASSUMES ABOUT YOUR REPO (github.com/Np2103/hyperking-src)
---------------------------------------------------------------
Based on what you've built so far, this script imports:

    patch_dataset.py      -> PatchDataset(patches_dir=..., ...)
    generator.py           -> GeneratorFirstHalf
                               .step_dnc(current_epoch, total_epochs, discriminator_loss=None)
    inverse_qc_module.py   -> InverseQCModule
    lowrank_module.py      -> LowRankModule
    noise_injector.py      -> NoiseInjector          (Stage 3, post-reconstruction)
    dnc_analyzer.py        -> DNCAnalyzer            (epsilon_1, epsilon_2, epsilon_3)
    ds_module.py           -> DSModule               (Discriminator start)
    he_quantum_classifier.py -> HEQuantumClassifier
    sigmoid_module.py      -> SigmoidModule
    losses.py              -> generator_loss(...), discriminator_loss(...)

>>> IMPORTANT: If any of these class/function names differ in your actual
>>> files (e.g. you called it "InverseQC" instead of "InverseQCModule"),
>>> just fix the import lines in the "IMPORTS FROM YOUR REPO" section below.
>>> Everything else (device handling, checkpointing, epoch loop, logging)
>>> does not depend on those exact names.

WHAT THIS SCRIPT ADDS ON TOP OF YOUR COLAB train.py
----------------------------------------------------
1. Proper CUDA device handling (.to(device) everywhere, works with 0 or
   many GPUs on the server, falls back to CPU with a warning if no GPU).
2. Command-line arguments instead of hardcoded Colab paths.
3. Checkpointing every N epochs (resumable — critical for a 2300-epoch
   run in case the server reboots, the SSH session dies, or you hit a
   time limit on a shared college server).
4. CSV loss logging (so you can plot G/D loss afterwards without re-running).
5. Graceful handling of SIGTERM (many HPC job schedulers send this before
   killing a job — this saves a checkpoint first instead of losing progress).
6. A configurable PennyLane device (lightning.qubit / lightning.gpu) since
   plain default.qubit is pure Python and is very likely your real
   speed bottleneck, not "CPU vs GPU" for the classical layers.

USAGE (see RUN_PROCEDURE.md for the full walkthrough)
------------------------------------------------------
    python3 train_gpu.py \\
        --data-dir /home/youruser/hyperking/patches \\
        --checkpoint-dir /home/youruser/hyperking/checkpoints \\
        --log-dir /home/youruser/hyperking/logs \\
        --epochs 2300 \\
        --batch-size 4 \\
        --alternate-period 5 \\
        --qdevice lightning.qubit

Resume after interruption (auto-detects the latest checkpoint):
    python3 train_gpu.py --data-dir ... --checkpoint-dir ... --resume
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
# IMPORTS FROM YOUR REPO  (clone hyperking-src next to this script, or
# add it to PYTHONPATH — see RUN_PROCEDURE.md step 3)
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
    print("IMPORT ERROR — a module from your hyperking-src repo was not found.")
    print(f"  Missing: {e}")
    print("Fix: either the class name in this script doesn't match your file,")
    print("     or the repo isn't on PYTHONPATH yet. See RUN_PROCEDURE.md step 3.")
    print("=" * 70)
    sys.exit(1)


# ----------------------------------------------------------------------
# Model wrappers — combine your existing building blocks into a full
# Generator and full Discriminator, mirroring what your Colab train.py
# already wires together. If your train.py builds these differently,
# copy YOUR wiring into these two classes instead — the rest of the
# script (training loop, checkpointing, device handling) stays the same.
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
        # First-half module already owns its own Stage 1 / Stage 2 DNC.
        self.first_half.step_dnc(current_epoch, total_epochs,
                                  discriminator_loss=discriminator_loss)
        self.stage3_dnc.step(current_epoch, total_epochs,
                              discriminator_loss=discriminator_loss)

    def forward(self, x):
        x = self.first_half(x)          # DC -> Reshape -> Core Quantum FE (+DNC stage1/2)
        x = self.inverse_qc(x)          # Inverse-QC
        x = self.low_rank(x)            # Low-rank -> 172x128x128
        if self.training:
            eps3 = self.stage3_dnc.get_epsilon_3()
            x = self.stage3_noise(x, epsilon=eps3)   # Novelty 1, Stage 3
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, qdevice="lightning.qubit"):
        super().__init__()
        self.ds = DSModule()
        self.he_classifier = HEQuantumClassifier(qdevice=qdevice)
        self.sigmoid = SigmoidModule()

    def forward(self, x):
        x = self.ds(x)
        x = self.he_classifier(x)
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


# ----------------------------------------------------------------------
# Graceful shutdown: if the server / job scheduler sends SIGTERM,
# save a checkpoint before the process is killed instead of losing progress.
# ----------------------------------------------------------------------

_shutdown_requested = {"flag": False}

def _handle_sigterm(signum, frame):
    print("\n[signal] SIGTERM received — will checkpoint at end of current epoch and exit.")
    _shutdown_requested["flag"] = True

signal.signal(signal.SIGTERM, _handle_sigterm)


# ----------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HyperKING GAN training on GPU server")
    parser.add_argument("--data-dir", required=True, help="Path to folder of .npy patches on THIS server")
    parser.add_argument("--checkpoint-dir", required=True, help="Where to save/resume checkpoints")
    parser.add_argument("--log-dir", default="./logs", help="Where to write loss_log.csv")
    parser.add_argument("--epochs", type=int, default=2300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5, help="RMSprop learning rate")
    parser.add_argument("--alternate-period", type=int, default=5,
                         help="Train G every epoch; train D every Nth epoch after warmup (matches base paper schedule)")
    parser.add_argument("--warmup-g-epochs", type=int, default=5,
                         help="Epochs to train G alone before alternating with D")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                         help="Save a checkpoint every N epochs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--qdevice", default="lightning.qubit",
                         choices=["default.qubit", "lightning.qubit", "lightning.gpu"],
                         help="PennyLane backend for the quantum modules. "
                              "lightning.qubit is usually the best CPU speedup with no extra setup. "
                              "lightning.gpu requires NVIDIA cuQuantum installed separately.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in --checkpoint-dir")
    parser.add_argument("--amp", action="store_true",
                         help="Use mixed precision for the classical (non-quantum) layers. "
                              "Leave off the first time — the quantum layers may not support autocast cleanly.")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ---------------- Device ----------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[device] WARNING: no CUDA GPU detected — training will run on CPU and be slow. "
              "Check `nvidia-smi` on the server and that torch was installed with CUDA support.")

    # ---------------- Data ----------------
    print(f"[data] Loading patches from {args.data_dir}")
    dataset = PatchDataset(patches_dir=args.data_dir)
    print(f"[data] {len(dataset)} patches found")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
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
            print("[resume] --resume was passed but no checkpoint found — starting fresh.")

    # ---------------- Logging ----------------
    log_path = os.path.join(args.log_dir, "loss_log.csv")
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["epoch", "g_loss", "d_loss", "epsilon_1", "epsilon_2", "epsilon_3", "seconds"])

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

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
            with torch.cuda.amp.autocast(enabled=args.amp):
                restored = generator(corrupted)
                d_pred_fake = discriminator(restored)
                g_loss = generator_loss(restored, clean, d_pred_fake)
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()

            d_loss_value = None
            # ---- Train Discriminator (only on alternation epochs) ----
            if train_d_this_epoch:
                opt_d.zero_grad()
                with torch.cuda.amp.autocast(enabled=args.amp):
                    d_pred_real = discriminator(clean)
                    d_pred_fake_detached = discriminator(restored.detach())
                    d_loss = discriminator_loss(d_pred_real, d_pred_fake_detached)
                scaler.scale(d_loss).backward()
                scaler.step(opt_d)
                scaler.update()
                d_loss_value = d_loss.item()
                running_d += d_loss_value

            running_g += g_loss.item()
            n_batches += 1

        avg_g = running_g / max(n_batches, 1)
        avg_d = (running_d / max(n_batches, 1)) if train_d_this_epoch else float("nan")
        elapsed = time.time() - epoch_start

        eps = generator.first_half.dnc if hasattr(generator.first_half, "dnc") else None
        eps1 = getattr(eps, "epsilon_1", None) if eps else None
        eps2 = getattr(eps, "epsilon_2", None) if eps else None
        eps3 = generator.stage3_dnc.get_epsilon_3() if hasattr(generator.stage3_dnc, "get_epsilon_3") else None

        print(f"[epoch {epoch:4d}/{args.epochs}] g_loss={avg_g:.5f} "
              f"d_loss={avg_d if train_d_this_epoch else 'skip':>8} "
              f"time={elapsed:.1f}s")

        log_writer.writerow([epoch, avg_g, avg_d, eps1, eps2, eps3, round(elapsed, 2)])
        log_file.flush()

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1 or _shutdown_requested["flag"]:
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
        print("TRAINING CRASHED — full traceback below. Your last checkpoint is safe")
        print("in --checkpoint-dir; re-run with --resume to continue from there.")
        print("=" * 70)
        traceback.print_exc()
        sys.exit(1)
