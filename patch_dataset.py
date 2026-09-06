"""
patch_dataset.py
-----------------
Handles everything needed to turn a raw .npy patch from data/patches/
into a (corrupted_input, clean_target) training pair ready for the Generator.

THREE things this file does:

  1. PREPROCESS  — fix the two problems found in your real patches:
       a) Fill value:  -50.0 (AVIRIS no-data marker) → clipped to 0
       b) Value range: raw reflectance (-50 to ~5000) → normalized to [0, 1]

  2. CORRUPT  — create the "damaged" version of the clean patch.
     Your project is about RESTORING corrupted hyperspectral images.
     The patches you have are the CLEAN satellite images (ground truth).
     To train the Generator you need pairs:
         input  = corrupted version  (what the Generator receives)
         target = clean version      (what the Generator must produce)
     Corruption simulates realistic AVIRIS satellite sensor damage:
       a) Gaussian noise       — general sensor noise
       b) Stripe noise         — dead/malfunctioning sensor columns
       c) Dead bands           — entire spectral bands that go dark

  3. DATASET CLASS  — a torch.utils.data.Dataset subclass that:
       - Scans data/patches/ for all .npy files
       - On each __getitem__ call: loads one patch, preprocesses, corrupts
       - Returns (corrupted_tensor, clean_tensor) ready for DataLoader

After building this, the FULL real-data pipeline looks like:

    DataLoader → (corrupted, clean) pair
        corrupted → Generator (DNC S1 → DC → Reshape → DNC S2 → QuantumFE)
        clean     → compared against Generator's final output via loss function

USAGE EXAMPLE (in your training loop on Colab):
    from patch_dataset import PatchDataset
    from torch.utils.data import DataLoader

    dataset = PatchDataset(patches_dir='data/patches')
    loader  = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for corrupted_batch, clean_batch in loader:
        # corrupted_batch: (4, 172, 128, 128) - Generator input
        # clean_batch:     (4, 172, 128, 128) - loss target
        generator.step_dnc(epoch, total_epochs)
        output = generator(corrupted_batch)
        loss   = criterion(output, clean_batch)
        ...
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants — tuned from inspecting your real patches
# ---------------------------------------------------------------------------

# Raw reflectance values in your AVIRIS patches go up to ~5000.
# We normalize by dividing by this value to bring everything into [0, 1].
# Why 5000 and not the exact max?
#   - Using a fixed constant (not per-patch max) keeps the scale CONSISTENT
#     across ALL patches and across train/eval/test splits. If you normalised
#     per-patch, a dark patch (max=1000) and a bright patch (max=5000) would
#     both be scaled to 1.0 — losing real scene brightness differences.
#   - 5000 is a safe ceiling: from inspection, real values stay under 5000.
REFLECTANCE_MAX = 5000.0

# AVIRIS "no data" fill value — pixels outside the sensor swath are set
# to exactly -50.0. These are NOT real measurements; clip them to 0.
FILL_VALUE = -50.0


# ---------------------------------------------------------------------------
# Step 1 — Preprocessing
# ---------------------------------------------------------------------------

def preprocess(patch_np: np.ndarray) -> torch.Tensor:
    """Convert a raw AVIRIS .npy patch to a normalised float tensor.

    Operations (in order):
      1. Clip: values below 0 → 0  (removes -50 fill value and tiny negatives)
      2. Clip: values above REFLECTANCE_MAX → REFLECTANCE_MAX  (safety ceiling)
      3. Normalise: divide by REFLECTANCE_MAX  → values in [0.0, 1.0]
      4. Convert to torch.float32 tensor

    Parameters
    ----------
    patch_np : np.ndarray
        Raw patch loaded from .npy file.
        Shape: (172, 128, 128),  dtype: float32 or float64
        Value range: approximately -50 to +5000

    Returns
    -------
    torch.Tensor
        Shape: (172, 128, 128),  dtype: torch.float32
        Value range: [0.0, 1.0]
    """
    # Work on a copy — don't modify the array loaded from disk
    arr = patch_np.astype(np.float32, copy=True)

    # Clip fill values and any other negatives to 0
    np.clip(arr, 0.0, REFLECTANCE_MAX, out=arr)

    # Normalise to [0, 1]
    arr /= REFLECTANCE_MAX

    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Step 2 — Corruption
# ---------------------------------------------------------------------------

class CorruptionEngine:
    """Applies realistic satellite sensor damage to a clean HSI patch.

    Three corruption types are applied together (each independently toggled
    via the constructor flags for ablation studies):

    1. Gaussian noise  — simulates general thermal/readout sensor noise.
       Each pixel gets an independent Gaussian perturbation.
       Strength controlled by `gaussian_sigma` (fraction of normalised range).

    2. Stripe noise    — simulates one or more dead/saturated sensor columns.
       Vertical stripes of either zeros or maximum value, spanning all 172 bands.
       The number and positions of stripes are randomly chosen each call.

    3. Dead bands      — simulates entire spectral channels going dark (sensor
       channel failures). The chosen bands are zeroed out completely.
       Number of dead bands randomly chosen from [0, max_dead_bands].

    Parameters
    ----------
    gaussian_sigma : float
        Standard deviation of Gaussian noise (in normalised [0,1] units).
        Default 0.03 means noise is ±3% of the full signal range — realistic
        for AVIRIS (typical SNR ~200:1 → noise ≈ 0.5% signal, but we use
        a slightly larger value to make restoration a meaningful challenge).
    stripe_prob : float
        Probability that any given column becomes a stripe in a call.
        Default 0.03 = ~4 stripes per 128-column image on average.
    max_dead_bands : int
        Upper bound on number of spectral bands killed per patch.
        Default 5 (out of 172 total bands). Set 0 to disable.
    use_gaussian : bool
        Whether to apply Gaussian noise. Default True.
    use_stripes : bool
        Whether to apply stripe noise. Default True.
    use_dead_bands : bool
        Whether to apply dead band corruption. Default True.
    """

    def __init__(
        self,
        gaussian_sigma: float = 0.03,
        stripe_prob: float = 0.03,
        max_dead_bands: int = 5,
        use_gaussian: bool = True,
        use_stripes: bool = True,
        use_dead_bands: bool = True,
    ):
        self.gaussian_sigma = gaussian_sigma
        self.stripe_prob = stripe_prob
        self.max_dead_bands = max_dead_bands
        self.use_gaussian = use_gaussian
        self.use_stripes = use_stripes
        self.use_dead_bands = use_dead_bands

    def corrupt(self, clean: torch.Tensor) -> torch.Tensor:
        """Apply corruption to a clean patch.

        Parameters
        ----------
        clean : torch.Tensor
            Preprocessed clean patch. Shape: (172, 128, 128), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Corrupted patch. Same shape (172, 128, 128), values clipped to [0, 1].
        """
        corrupted = clean.clone()
        bands, height, width = corrupted.shape

        # --- Gaussian noise ---
        if self.use_gaussian and self.gaussian_sigma > 0:
            noise = torch.randn_like(corrupted) * self.gaussian_sigma
            corrupted = corrupted + noise

        # --- Stripe noise (dead/saturated columns) ---
        if self.use_stripes and self.stripe_prob > 0:
            for col in range(width):
                if random.random() < self.stripe_prob:
                    # 50% chance: dead column (→ 0), 50% chance: saturated (→ 1)
                    fill = 0.0 if random.random() < 0.5 else 1.0
                    corrupted[:, :, col] = fill  # all bands, all rows, this column

        # --- Dead bands ---
        if self.use_dead_bands and self.max_dead_bands > 0:
            n_dead = random.randint(0, self.max_dead_bands)
            if n_dead > 0:
                dead_band_indices = random.sample(range(bands), n_dead)
                for b in dead_band_indices:
                    corrupted[b, :, :] = 0.0

        # Clip back to valid range [0, 1] after noise addition
        corrupted = corrupted.clamp(0.0, 1.0)

        return corrupted


# ---------------------------------------------------------------------------
# Step 3 — Dataset class
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """Dataset of (corrupted, clean) AVIRIS hyperspectral patch pairs.

    Scans `patches_dir` for all .npy files, preprocesses and corrupts them
    on-the-fly during training. No preprocessing is done ahead of time —
    each call to __getitem__ loads, preprocesses, and corrupts one patch.

    This design keeps disk usage low (only the raw .npy files are stored)
    and gives fresh, randomly-corrupted patches on every epoch.

    Parameters
    ----------
    patches_dir : str
        Path to the directory containing .npy patch files.
        Example: 'data/patches'
    corruption : CorruptionEngine, optional
        An instance of CorruptionEngine. If None, a default one is created.
        Pass a custom instance to change corruption settings.
    split : str
        One of 'train', 'val', 'test'. Controls which fraction of the
        1,062 patches this Dataset uses:
            train : first 80%  (~850 patches)
            val   : next  10%  (~106 patches)
            test  : last  10%  (~106 patches)
        Files are sorted alphabetically before splitting (reproducible).
    seed : int
        Random seed used for the train/val/test split shuffling.
        Default 42. Change only if you want a different split.
    """

    SPLIT_FRACTIONS = {
        'train': (0.0, 0.80),
        'val':   (0.80, 0.90),
        'test':  (0.90, 1.00),
    }

    def __init__(
        self,
        patches_dir: str = os.path.join('data', 'patches'),
        corruption: CorruptionEngine = None,
        split: str = 'train',
        seed: int = 42,
    ):
        super().__init__()

        if split not in self.SPLIT_FRACTIONS:
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got '{split}'"
            )

        self.patches_dir = patches_dir
        self.corruption = corruption if corruption is not None else CorruptionEngine()
        self.split = split

        # Discover all .npy files
        all_files = sorted(glob.glob(os.path.join(patches_dir, '*.npy')))
        if not all_files:
            raise FileNotFoundError(
                f"No .npy files found in '{patches_dir}'. "
                f"Check the path. Expected location: data/patches/"
            )

        # Deterministic shuffle then split
        rng = random.Random(seed)
        rng.shuffle(all_files)

        lo_frac, hi_frac = self.SPLIT_FRACTIONS[split]
        n = len(all_files)
        lo_idx = int(n * lo_frac)
        hi_idx = int(n * hi_frac)
        self.files = all_files[lo_idx:hi_idx]

        if len(self.files) == 0:
            raise RuntimeError(
                f"Split '{split}' resulted in 0 files from {n} total. "
                f"Check split fractions."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        """Load, preprocess, and corrupt one patch.

        Parameters
        ----------
        idx : int
            Index into self.files.

        Returns
        -------
        corrupted : torch.Tensor  shape (172, 128, 128)  values in [0, 1]
            The damaged version — this is what the Generator receives.
        clean : torch.Tensor  shape (172, 128, 128)  values in [0, 1]
            The original clean version — this is the training target.
        """
        patch_path = self.files[idx]
        patch_np = np.load(patch_path)          # (172, 128, 128), raw values
        clean = preprocess(patch_np)            # (172, 128, 128), [0, 1]
        corrupted = self.corruption.corrupt(clean)  # (172, 128, 128), [0, 1]
        return corrupted, clean

    def __repr__(self) -> str:
        return (
            f"PatchDataset(split='{self.split}', "
            f"n_patches={len(self.files)}, "
            f"patches_dir='{self.patches_dir}')"
        )


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly:
#   python src/patch_dataset.py
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    print('=' * 60)
    print('PatchDataset — real patch pipeline sanity check')
    print('=' * 60)

    # ── Step 1: Preprocess a single raw patch ──────────────────────
    print('\n--- Step 1: Preprocessing ---')
    raw_path = os.path.join('data', 'patches',
                            'f080611t01p00r06rdn_c_sc01_ort_img_r0000_c0000.npy')
    raw = np.load(raw_path)
    print(f'Raw patch — shape: {raw.shape}, min: {raw.min():.1f}, max: {raw.max():.1f}')
    clean = preprocess(raw)
    print(f'After preprocess — shape: {tuple(clean.shape)}, '
          f'min: {clean.min():.4f}, max: {clean.max():.4f}')
    neg_count = (clean < 0).sum().item()
    over1_count = (clean > 1).sum().item()
    print(f'Values < 0: {neg_count}   (must be 0)')
    print(f'Values > 1: {over1_count}  (must be 0)')
    assert neg_count == 0 and over1_count == 0, \
        'Preprocessing failed: values outside [0, 1]'
    print('Preprocessing: PASSED')

    # ── Step 2: Corruption ─────────────────────────────────────────
    print('\n--- Step 2: Corruption ---')
    engine = CorruptionEngine(
        gaussian_sigma=0.03,
        stripe_prob=0.03,
        max_dead_bands=5,
    )
    corrupted = engine.corrupt(clean)
    print(f'Clean    — min: {clean.min():.4f}, max: {clean.max():.4f}, '
          f'mean: {clean.mean():.4f}')
    print(f'Corrupted — min: {corrupted.min():.4f}, max: {corrupted.max():.4f}, '
          f'mean: {corrupted.mean():.4f}')
    diff = (clean - corrupted).abs()
    print(f'Mean absolute difference (clean vs corrupted): {diff.mean():.6f}')
    print(f'(Should be > 0 — confirms corruption was applied)')
    assert diff.mean() > 0, 'Corruption had no effect!'
    assert corrupted.min() >= 0.0 and corrupted.max() <= 1.0, \
        'Corrupted values went outside [0, 1]'
    print('Corruption: PASSED')

    # ── Step 3: Dataset ─────────────────────────────────────────────
    print('\n--- Step 3: PatchDataset ---')
    train_ds = PatchDataset(patches_dir=os.path.join('data', 'patches'),
                            split='train')
    val_ds   = PatchDataset(patches_dir=os.path.join('data', 'patches'),
                            split='val')
    test_ds  = PatchDataset(patches_dir=os.path.join('data', 'patches'),
                            split='test')

    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f'Train: {len(train_ds)} patches')
    print(f'Val:   {len(val_ds)} patches')
    print(f'Test:  {len(test_ds)} patches')
    print(f'Total: {total} patches (all .npy files in data/patches/)')

    # Load one item and check shapes
    print('\nLoading item 0 from train split...')
    t0 = time.time()
    c_patch, clean_patch = train_ds[0]
    elapsed = time.time() - t0
    print(f'  corrupted shape: {tuple(c_patch.shape)}   (Generator INPUT)')
    print(f'  clean shape:     {tuple(clean_patch.shape)} (loss TARGET)')
    print(f'  corrupted range: [{c_patch.min():.4f}, {c_patch.max():.4f}]')
    print(f'  clean range:     [{clean_patch.min():.4f}, {clean_patch.max():.4f}]')
    print(f'  Load+preprocess+corrupt time: {elapsed:.2f}s')
    assert c_patch.shape == (172, 128, 128)
    assert clean_patch.shape == (172, 128, 128)
    assert c_patch.dtype == torch.float32
    print('PatchDataset item shapes: PASSED')

    # ── Step 4: DataLoader batch ────────────────────────────────────
    print('\n--- Step 4: DataLoader (batch_size=2) ---')
    loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    c_batch, clean_batch = next(iter(loader))
    print(f'  Batch corrupted: {tuple(c_batch.shape)}  (batch of 2 Generator inputs)')
    print(f'  Batch clean:     {tuple(clean_batch.shape)}  (batch of 2 targets)')
    assert c_batch.shape == (2, 172, 128, 128)
    assert clean_batch.shape == (2, 172, 128, 128)
    print('DataLoader batch: PASSED')

    # ── Step 5: Full pipeline with Generator ────────────────────────
    print('\n--- Step 5: Real patch -> Full Generator pipeline ---')
    print('(Quantum circuit runs 128x2 evaluations -- may take ~60s on CPU)')

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from generator import GeneratorFirstHalf

    gen = GeneratorFirstHalf(in_channels=172)
    gen.eval()  # eval mode: no DNC noise (we add our own corruption above)

    one_corrupted = c_batch[:1]  # take 1 patch from batch: (1, 172, 128, 128)
    print(f'  Input  -> Generator:  {tuple(one_corrupted.shape)}  real AVIRIS patch')
    t0 = time.time()
    with torch.no_grad():
        output = gen(one_corrupted)
    elapsed = time.time() - t0
    print(f'  Output <- Generator: {tuple(output.shape)}  (expected: (1, 64, 2, 2))')
    print(f'  Time: {elapsed:.1f}s')
    assert output.shape == (1, 64, 2, 2), \
        f'Wrong output shape: {output.shape}'
    print('Real patch -> Generator pipeline: PASSED')

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('ALL CHECKS PASSED')
    print()
    print('Your real AVIRIS patches are now flowing through the pipeline:')
    print('  data/patches/*.npy')
    print('    -> preprocess (clip + normalise to [0,1])')
    print('    -> corrupt (Gaussian + stripes + dead bands)')
    print('    -> Generator (DNC S1 -> DC -> Reshape -> DNC S2 -> QuantumFE)')
    print('    -> (1, 64, 2, 2) quantum feature map')
    print()
    print('Next steps (on Colab):')
    print('  -> Build Inverse-QC module  64x2x2 -> 8x128x128')
    print('  -> Build Low-rank module    8x128x128 -> 172x128x128')
    print('  -> Connect loss function and training loop')
    print('=' * 60)
