#!/bin/bash
# ============================================================
# 04_evaluate.sh — Evaluate all checkpoints, produce JSON rows
# ============================================================
# Runs evaluate.py on every final checkpoint from steps 02 and 03.
# Each evaluation produces one JSON file with PPL, joules, entropy.
#
# Estimated time: ~10-20 min total (evaluation is faster than training)
#
# Outputs:
#   results/m2/switch/eval.json
#   results/m2/random/eval.json
#   results/m2/dense/eval.json
#   results/m2/energy_aware/lambda_*/eval.json
#   results/m2/all_evals.jsonl          ← combined, one row per run
#
# Usage:
#   bash scripts/runpod/04_evaluate.sh 2>&1 | tee logs/04_evaluate.log
# ============================================================
set -euo pipefail

echo "============================================"
echo " Step 4: Evaluation"
echo " $(date)"
echo "============================================"

mkdir -p logs

# Collect all final checkpoints
COMBINED="results/m2/all_evals.jsonl"
> "$COMBINED"  # truncate

evaluate_checkpoint() {
    local ckpt=$1
    local output=$2
    local label=$3

    if [ ! -f "$ckpt" ]; then
        echo "  SKIP: $ckpt not found"
        return
    fi

    echo ""
    echo "--- Evaluating: $label ---"
    python3 evaluate.py \
        --checkpoint "$ckpt" \
        --data-dir ./data/processed/c4_1pct \
        --output "$output" \
        --batch-size 64

    # Append to combined file (flatten JSON to single line)
    python3 -c "
import json
with open('$output') as f:
    data = json.load(f)
print(json.dumps(data))
" >> "$COMBINED"

    echo "  Saved: $output"
}

# ── Baselines ──
evaluate_checkpoint \
    "results/m2/switch/checkpoints/final.pt" \
    "results/m2/switch/eval.json" \
    "Switch (lambda=0)"

evaluate_checkpoint \
    "results/m2/random/checkpoints/final.pt" \
    "results/m2/random/eval.json" \
    "Random"

evaluate_checkpoint \
    "results/m2/dense/checkpoints/final.pt" \
    "results/m2/dense/eval.json" \
    "Dense"

# ── Energy-aware sweep ──
for LAMBDA_DIR in results/m2/energy_aware/lambda_*/; do
    if [ -d "$LAMBDA_DIR" ]; then
        LAMBDA_VAL=$(basename "$LAMBDA_DIR" | sed 's/lambda_//')
        evaluate_checkpoint \
            "${LAMBDA_DIR}checkpoints/final.pt" \
            "${LAMBDA_DIR}eval.json" \
            "Energy-aware (lambda=$LAMBDA_VAL)"
    fi
done

# ── Summary table ──
echo ""
echo "============================================"
echo " Evaluation Summary"
echo "============================================"

python3 -c "
import json

print(f'{\"Router\":>15s}  {\"Lambda\":>7s}  {\"PPL\":>8s}  {\"Joules\":>10s}  {\"Entropy\":>8s}  {\"Balance\":>8s}')
print('-' * 65)

with open('$COMBINED') as f:
    for line in f:
        d = json.loads(line)
        lam = d.get('lambda_energy', 0.0)
        lam_str = f'{lam:.2f}' if d['router_type'] == 'energy_aware' else '-'
        print(f'{d[\"router_type\"]:>15s}  {lam_str:>7s}  {d[\"ppl\"]:>8.2f}  {d[\"total_joules\"]:>10.1f}  {d[\"mean_router_entropy\"]:>8.3f}  {d.get(\"balance_loss\", 0):>8.4f}')
"

echo ""
echo "Combined results: $COMBINED"
echo ""
echo "============================================"
echo " Evaluation complete. Run 05_plot.sh next."
echo " $(date)"
echo "============================================"
