# Running HyperKING Training on Your College GPU Server — Full Procedure

You're moving from Colab (a notebook, with Drive auto-mounted) to a real
server (a plain Linux machine you SSH into, nothing auto-mounted, nothing
pre-installed). Here is every step, in order.

---

## Step 0 — Get access details from your college

Ask your admin/lab for:
- The server's hostname or IP address
- Your username
- How you log in (password, or an SSH key file they give you)
- Whether you use it directly, or submit jobs through a queue (SLURM/PBS).
  If they mention "sbatch" or "qsub", tell me — the run command changes
  slightly (see the note at the bottom).

---

## Step 1 — Connect to the server

From your laptop (Windows: use PowerShell, or PuTTY if they gave you a `.ppk` key):

```bash
ssh yourusername@server-hostname-or-ip
```

If they gave you a key file:
```bash
ssh -i /path/to/keyfile.pem yourusername@server-hostname-or-ip
```

You're now typing commands *on the server*, not your laptop.

---

## Step 2 — Copy the 4 files I've given you onto the server

From a **new terminal on your laptop** (keep the SSH session open in the other one):

```bash
scp train_gpu.py requirements_gpu.txt setup_gpu_server.sh RUN_PROCEDURE.md yourusername@server-hostname-or-ip:~/
```

---

## Step 3 — Run the setup script (one time only)

Back in your SSH session:

```bash
chmod +x setup_gpu_server.sh
./setup_gpu_server.sh
```

This will:
1. Confirm the GPU is visible (`nvidia-smi`)
2. Create a Python virtual environment
3. Clone your `hyperking-src` GitHub repo
4. Ask you to paste the correct PyTorch install command for the server's
   CUDA version (it prints the driver version and a link — pytorch.org
   gives you the exact command, just copy-paste it in when prompted)
5. Install the rest of the requirements (PennyLane, etc.)
6. Print a confirmation that `torch.cuda.is_available()` is `True`

**If step 6 prints `False`:** stop and don't start training — that means
PyTorch installed the CPU-only version. Re-run `pip install torch ...`
with the correct `--index-url` for your CUDA version.

---

## Step 4 — Move `train_gpu.py` into the repo folder

Your imports (`from generator import GeneratorFirstHalf` etc.) only work if
`train_gpu.py` sits next to your module files, or the repo folder is on
`PYTHONPATH`. Simplest fix:

```bash
cp ~/train_gpu.py ~/hyperking_project/hyperking-src/
cd ~/hyperking_project/hyperking-src/
```

---

## Step 5 — Get your 1366 patches onto the server (no Drive mount here)

Pick **one** of these:

**Option A — Direct transfer from your laptop (simplest if patches are still there)**
```bash
# run this from your LAPTOP, not the server
scp -r "C:\path\to\your\patches" yourusername@server-hostname-or-ip:~/hyperking_project/patches
```
(On Windows PowerShell, `scp` works the same way if you have OpenSSH installed, which it does by default on Windows 10/11.)

**Option B — Download from your Google Drive folder directly on the server**
1. On Drive, right-click the `Hyperking/patches` folder → Share → "Anyone with the link"
2. Get the folder ID from the link
3. On the server:
   ```bash
   pip install gdown
   gdown --folder https://drive.google.com/drive/folders/YOUR_FOLDER_ID -O ~/hyperking_project/patches
   ```

**Option C — Re-run extraction on the server**
If you still have the raw AVIRIS scenes, copy those instead (`scp -r`) and
re-run your `extract_patches.py` on the server — avoids transferring
already-large patch files twice.

Whichever you pick, confirm the count matches:
```bash
ls ~/hyperking_project/patches | wc -l
```
Should show 1366 (or 1450 if you've since fixed the Drive storage limit
and moved the rest over).

---

## Step 6 — Start training

Still inside `~/hyperking_project/hyperking-src/` with the venv active
(`source ~/hyperking_project/venv/bin/activate` if you opened a new shell):

```bash
python3 train_gpu.py \
    --data-dir ~/hyperking_project/patches \
    --checkpoint-dir ~/hyperking_project/checkpoints \
    --log-dir ~/hyperking_project/logs \
    --epochs 2300 \
    --batch-size 4 \
    --qdevice lightning.qubit
```

### Important: run this inside `tmux` so it survives you closing your laptop

SSH sessions die the moment your laptop sleeps or loses wifi — and a
2300-epoch run will take hours. Wrap it in `tmux`:

```bash
tmux new -s hyperking_train
# (now you're inside a persistent session)
python3 train_gpu.py --data-dir ~/hyperking_project/patches \
    --checkpoint-dir ~/hyperking_project/checkpoints \
    --log-dir ~/hyperking_project/logs --epochs 2300 --batch-size 4 \
    --qdevice lightning.qubit
```
Then detach (training keeps running): press `Ctrl+B`, then `D`.
Close your laptop, go do something else.

To check on it later, SSH back in and run:
```bash
tmux attach -t hyperking_train
```

---

## Step 7 — Monitor progress

In a second SSH session (or another tmux pane):
```bash
watch -n 2 nvidia-smi          # confirm the GPU is actually busy
tail -f ~/hyperking_project/logs/loss_log.csv    # watch losses update live
```

---

## Step 8 — If it crashes or the server reboots

Nothing is lost past your last checkpoint (saved every 25 epochs by
default). Resume with:

```bash
python3 train_gpu.py \
    --data-dir ~/hyperking_project/patches \
    --checkpoint-dir ~/hyperking_project/checkpoints \
    --log-dir ~/hyperking_project/logs \
    --epochs 2300 --batch-size 4 --qdevice lightning.qubit \
    --resume
```

---

## Step 9 — Get your trained checkpoints back to your laptop

```bash
# from your LAPTOP
scp -r yourusername@server-hostname-or-ip:~/hyperking_project/checkpoints ./checkpoints
scp yourusername@server-hostname-or-ip:~/hyperking_project/logs/loss_log.csv ./loss_log.csv
```

---

## About speed: GPU vs. the quantum layers

Your classical layers (DC, Reshape, Inverse-QC, Low-rank, DS Module,
Sigmoid) will genuinely speed up on the GPU — that part of "switch to a
real GPU server" is straightforward and `train_gpu.py` already does
`.to(device)` for you.

**But** the Core Quantum FE and HE Quantum Classifier run on PennyLane
quantum simulators, and PennyLane's default device (`default.qubit`) is
pure Python and runs on **CPU regardless of what GPU you have** — that's
almost certainly what you meant by "slow per-group PennyLane circuit
execution" earlier. Two levers, in order of how easy they are:

1. **`--qdevice lightning.qubit`** (default in this script) — a much
   faster CPU backend, drop-in replacement, no extra install beyond
   `pip install pennylane-lightning`, usually the single biggest win for
   a small 4-qubit circuit like yours.
2. **`--qdevice lightning.gpu`** — actually uses the GPU for the quantum
   simulation, but needs NVIDIA's cuQuantum SDK installed on the server
   separately (`pip install pennylane-lightning[gpu]` plus the CUDA
   toolkit's cuQuantum libraries). Worth asking your GPU server's admin
   if cuQuantum is already installed — if not, it's extra setup you may
   not have time for today, so `lightning.qubit` is the safe default to
   start with right now.

---

## If your college server uses a job scheduler (SLURM/PBS) instead of direct SSH access

Tell me and I'll write you a matching `.slurm` or `.pbs` job script —
the difference is you'd submit the job with `sbatch run_job.slurm`
instead of running `python3 train_gpu.py` directly, and the scheduler
queues it for you rather than running it in your live SSH session.
