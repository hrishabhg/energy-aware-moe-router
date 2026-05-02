#!/bin/bash
# ============================================================
# 02_baselines.sh — Train all three baselines
# ============================================================
# Runs Switch, Random, and Dense baselines sequentially.
# Each run: ~50M tokens on the 1% C4 subsample.
#
# Estimated time per run on A100: ~30-60 min (16M param model)
# Total: ~2-3 hours for all three.
#
# Outputs:
#   results/m2/switch/    → checkpoints/, metrics.jsonl, config.yaml
#   results/m2/random/    → checkpoints/, metrics.jsonl, config.yaml
#   results/m2/dense/     → checkpoints/, metrics.jsonl, config.yaml
#
# Usage:
#   bash scripts/runpod/02_baselines.sh 2>&1 | tee logs/02_baselines.log
#
# To run a single baseline (e.g., for debugging):
#   bash scripts/runpod/02_baselines.sh switch
# ============================================================
set -euo pipefail

echo "============================================"
echo " Step 2: Baseline Training"
echo " $(date)"
echo "============================================"

# Check data exists
if [ ! -d "./data/processed/c4_1pct" ]; then
    echo "ERROR: C4 data not found at ./data/processed/c4_1pct"
    echo "Run 01_data.sh first."
    exit 1
fi

mkdir -p logs results/m2

SINGLE="${1:-all}"

run_baseline() {
    local router=$1
    echo ""
    echo "============================================"
    echo " Training: --router $router"
    echo " Started: $(date)"
    echo "============================================"

    python3 train.py \
        --config configs/tiny_moe.yaml \
        --router "$router" \
        --seed 42

    echo ""
    echo " Finished: --router $router at $(date)"
    echo "============================================"
}

if [ "$SINGLE" = "all" ]; then
    # ── Baseline 1: Switch Transformer (primary baseline) ──
    run_baseline "switch"

    # ── Baseline 2: Random Router (sanity check) ──
    run_baseline "random"

    # ── Baseline 3: Dense Transformer (energy reference) ──
    run_baseline "dense"
else
    run_baseline "$SINGLE"
fi

# ── Sanity checks ──
echo ""
echo "============================================"
echo " Baseline Sanity Checks"
echo "============================================"

python3 -c "
import json, sys

def read_last_metrics(path):
    with open(path) as f:
        lines = f.readlines()
    return json.loads(lines[-1]) if lines else {}

routers = []
for r in ['switch', 'random', 'dense']:
    try:
        m = read_last_metrics(f'results/m2/{r}/metrics.jsonl')
        routers.append((r, m))
        print(f'{r:>8s}:  loss={m[\"loss\"]:.4f}  ppl={m[\"ppl\"]:.2f}  entropy={m.get(\"mean_router_entropy\", 0):.3f}')
    except FileNotFoundError:
        print(f'{r:>8s}:  NOT FOUND (skipped or not yet run)')

# Sanity: Switch should beat Random on PPL
if len(routers) >= 2:
    switch_ppl = next((m['ppl'] for r, m in routers if r == 'switch'), None)
    random_ppl = next((m['ppl'] for r, m in routers if r == 'random'), None)
    if switch_ppl and random_ppl:
        if switch_ppl < random_ppl:
            print(f'\nSANITY OK: Switch PPL ({switch_ppl:.2f}) < Random PPL ({random_ppl:.2f})')
        else:
            print(f'\nWARNING: Switch PPL ({switch_ppl:.2f}) >= Random PPL ({random_ppl:.2f})')
            print('This suggests a bug in the MoE container. Debug before proceeding.')
"

echo ""
echo "============================================"
echo " Baselines complete. Run 03_energy_sweep.sh next."
echo " $(date)"
echo "============================================"
