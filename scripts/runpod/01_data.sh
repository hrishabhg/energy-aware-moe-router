#!/bin/bash
# ============================================================
# 01_data.sh — Download C4 1% subsample + smoke test
# ============================================================
# Downloads ~3.6M examples via streaming (no 750GB full download).
# Takes 2-4 hours. Run once; data persists if you Stop the pod.
#
# For a quick test first (5 min), run with --quick flag:
#   bash scripts/runpod/01_data.sh --quick
#
# Usage:
#   bash scripts/runpod/01_data.sh 2>&1 | tee logs/01_data.log
# ============================================================
set -euo pipefail

echo "============================================"
echo " Step 1: Data Download"
echo " $(date)"
echo "============================================"

QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
    echo "*** Quick mode: downloading 10K examples only ***"
fi

mkdir -p logs data/processed

# ── Download ──
if [ "$QUICK" = true ]; then
    echo "--- Downloading quick test subset (10K examples) ---"
    python3 scripts/download_c4.py \
        --streaming-subsample \
        --ratio 0.01 \
        --seed 42 \
        --max-examples 10000 \
        --output-dir ./data/processed/c4_quick

    DATA_DIR="./data/processed/c4_quick"
else
    if [ -d "./data/processed/c4_1pct/dataset_info.json" ] || [ -d "./data/processed/c4_1pct" ]; then
        echo "--- c4_1pct already exists, checking ---"
        python3 -c "
from datasets import load_from_disk
ds = load_from_disk('./data/processed/c4_1pct')
print(f'Existing dataset: {len(ds):,} examples')
if len(ds) > 100_000:
    print('Looks complete. Skipping download.')
else:
    print('Too small. Re-downloading.')
    exit(1)
" && { echo "Skipping download."; DATA_DIR="./data/processed/c4_1pct"; } || {
            echo "--- Downloading full 1% subsample (~3.6M examples) ---"
            echo "This will take 2-4 hours. Go do something else."
            python3 scripts/download_c4.py \
                --streaming-subsample \
                --ratio 0.01 \
                --seed 42 \
                --output-dir ./data/processed/c4_1pct
            DATA_DIR="./data/processed/c4_1pct"
        }
    else
        echo "--- Downloading full 1% subsample (~3.6M examples) ---"
        echo "This will take 2-4 hours. Go do something else."
        python3 scripts/download_c4.py \
            --streaming-subsample \
            --ratio 0.01 \
            --seed 42 \
            --output-dir ./data/processed/c4_1pct
        DATA_DIR="./data/processed/c4_1pct"
    fi
    DATA_DIR="./data/processed/c4_1pct"
fi

# ── Smoke test ──
echo ""
echo "--- Smoke testing data loader ---"
python3 src/data_loader.py \
    --arrow-dir "$DATA_DIR" \
    --batch-size 4 \
    --seq-len 512 \
    --num-batches 5

echo ""
echo "--- Data stats ---"
python3 -c "
from datasets import load_from_disk
ds = load_from_disk('$DATA_DIR')
print(f'Examples:  {len(ds):,}')
total_chars = sum(len(ex['text']) for ex in ds.select(range(min(1000, len(ds)))))
avg_chars = total_chars / min(1000, len(ds))
print(f'Avg chars: {avg_chars:.0f} per document (sampled from first 1000)')
"

echo ""
echo "============================================"
echo " Data ready at: $DATA_DIR"
echo " Run 02_baselines.sh next."
echo " $(date)"
echo "============================================"
