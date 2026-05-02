#!/bin/bash
# ============================================================
# 00_setup.sh — One-time RunPod environment setup
# ============================================================
# Run this ONCE when you first create the pod.
# After this, if you Stop/Start the pod, skip to 01_data.sh
# (the environment persists on disk).
#
# Template: PyTorch 2.x (RunPod pre-built)
# GPU: A100 40GB (or 80GB)
# Disk: 50 GB (enough for env + 1% C4 subsample + checkpoints)
#
# Usage:
#   cd /workspace
#   git clone <your-repo-url> energy-aware-moe-router
#   cd energy-aware-moe-router
#   bash scripts/runpod/00_setup.sh 2>&1 | tee logs/00_setup.log
# ============================================================
set -euo pipefail

echo "============================================"
echo " Step 0: Environment Setup"
echo " $(date)"
echo "============================================"

# Create logs directory
mkdir -p logs

# ── Check GPU ──
echo ""
echo "--- GPU Check ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Install Python packages ──
# RunPod PyTorch template already has torch + CUDA.
# We just need the project-specific packages.
echo "--- Installing project dependencies ---"
pip install --quiet \
    transformers>=4.46.0 \
    tokenizers>=0.20.0 \
    sentencepiece>=0.2.0 \
    "datasets>=3.0.0,<4.0.0" \
    evaluate>=0.4.0 \
    wandb>=0.18.0 \
    omegaconf>=2.3.0 \
    rich>=13.0.0 \
    nvidia-ml-py>=12.560.30 \
    pynvml>=11.5.0 \
    pyyaml>=6.0 \
    scipy>=1.12.0 \
    seaborn>=0.13.0

echo ""
echo "--- Verifying imports ---"
python3 -c "
import torch
print(f'PyTorch:     {torch.__version__}')
print(f'CUDA avail:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:         {torch.cuda.get_device_name(0)}')
    print(f'VRAM:        {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import transformers
print(f'Transformers:{transformers.__version__}')

import datasets
print(f'Datasets:    {datasets.__version__}')

import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
print(f'pynvml OK:   current power = {power:.1f} W')
pynvml.nvmlShutdown()
print()
print('All imports verified.')
"

echo ""
echo "--- Testing EnergyMonitor ---"
python3 -c "
import sys, time
sys.path.insert(0, '.')
from src.utils.energy_monitor import EnergyMonitor
mon = EnergyMonitor(device_indices=[0], sampling_hz=10)
mon.start()
time.sleep(2)
w = mon.latest_power(0)
j = mon.total_joules(0)
summary = mon.stop()
print(f'Power: {w:.1f} W, Energy: {j:.1f} J over 2s')
print(f'Overhead: {summary[\"sampling_overhead_pct\"]:.4f}%')
print('EnergyMonitor test passed.')
"

echo ""
echo "============================================"
echo " Setup complete. Run 01_data.sh next."
echo " $(date)"
echo "============================================"
