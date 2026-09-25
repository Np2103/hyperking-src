#!/bin/bash
# ============================================================
# setup_gpu_server.sh
# Run this ONCE after you SSH into your college GPU server.
# It sets up everything needed to run train_gpu.py.
# ============================================================
set -e   # stop immediately if any command fails

echo "== Step 1: Check the GPU is visible =="
nvidia-smi || { echo "nvidia-smi failed — ask the admin whether you have GPU access on this login node/queue."; exit 1; }

echo "== Step 2: Create a project folder and Python virtual environment =="
mkdir -p ~/hyperking_project
cd ~/hyperking_project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo "== Step 3: Clone your code repo =="
if [ ! -d "hyperking-src" ]; then
    git clone https://github.com/Np2103/hyperking-src.git
else
    echo "hyperking-src already exists, pulling latest..."
    (cd hyperking-src && git pull)
fi

echo "== Step 4: Install PyTorch with CUDA support =="
echo "Checking CUDA version reported by the driver..."
nvidia-smi --query-gpu=driver_version,name --format=csv
echo ""
echo ">>> Go to https://pytorch.org/get-started/locally/ , pick your CUDA version,"
echo ">>> and it will give you the exact pip command. Typically it looks like:"
echo ""
echo "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
echo ""
read -p "Paste that command here and press Enter to run it now: " TORCH_CMD
eval "$TORCH_CMD"

echo "== Step 5: Install the rest of the requirements =="
pip install -r requirements_gpu.txt

echo "== Step 6: Verify PyTorch sees the GPU =="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo ""
echo "Setup complete. Next: transfer your patches data (see RUN_PROCEDURE.md step 5),"
echo "then run training (RUN_PROCEDURE.md step 6)."
