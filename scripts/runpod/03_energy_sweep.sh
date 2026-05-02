#!/bin/bash
# ============================================================
# 03_energy_sweep.sh — Lambda sweep for the Pareto frontier
# ============================================================
# Trains 7 energy-aware models with different lambda values.
# Each produces one (PPL, joules) point on the Pareto plot.
#
# Lambda values: 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0
# (lambda=0 is already the Switch baseline from step 02)
#
# Estimated time per run on A100: ~30-60 min
# Total: ~4-7 hours for the full sweep.
#
# Outputs:
#   results/m2/energy_aware/lambda_0.01/
#   results/m2/energy_aware/lambda_0.05/
#   ...etc
#
# To run a single lambda (e.g., for testing):
#   bash scripts/runpod/03_energy_sweep.sh 0.1
#
# Usage:
#   bash scripts/runpod/03_energy_sweep.sh 2>&1 | tee logs/03_energy_sweep.log
# ============================================================
set -euo pipefail

echo "============================================"
echo " Step 3: Energy-Aware Lambda Sweep"
echo " $(date)"
echo "============================================"

# Check prerequisites
if [ ! -f "results/m2/switch/checkpoints/final.pt" ]; then
    echo "WARNING: Switch baseline not found. Run 02_baselines.sh first."
    echo "Proceeding anyway (the sweep doesn't depend on baseline checkpoints)."
fi

mkdir -p logs

# Lambda values to sweep
LAMBDAS="${1:-0.01 0.05 0.1 0.25 0.5 1.0 2.0}"

for LAMBDA in $LAMBDAS; do
    echo ""
    echo "============================================"
    echo " Training: lambda_energy = $LAMBDA"
    echo " Started: $(date)"
    echo "============================================"

    python3 train.py \
        --config configs/tiny_moe.yaml \
        --router energy_aware \
        --lambda-energy "$LAMBDA" \
        --seed 42

    echo " Finished: lambda=$LAMBDA at $(date)"
done

echo ""
echo "============================================"
echo " Lambda sweep summary"
echo "============================================"

python3 -c "
import json, os

print(f'{\"lambda\":>8s}  {\"loss\":>8s}  {\"ppl\":>8s}  {\"entropy\":>8s}')
print('-' * 40)

for lam in ['0.01', '0.05', '0.1', '0.25', '0.5', '1.0', '2.0']:
    path = f'results/m2/energy_aware/lambda_{lam}/metrics.jsonl'
    if not os.path.exists(path):
        continue
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        continue
    m = json.loads(lines[-1])
    print(f'{lam:>8s}  {m[\"loss\"]:>8.4f}  {m[\"ppl\"]:>8.2f}  {m.get(\"mean_router_entropy\", 0):>8.3f}')
"

echo ""
echo "============================================"
echo " Sweep complete. Run 04_evaluate.sh next."
echo " $(date)"
echo "============================================"
