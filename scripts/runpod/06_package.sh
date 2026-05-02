#!/bin/bash
# ============================================================
# 06_package.sh — Bundle results for download before terminating
# ============================================================
# Creates a tar.gz of all results, figures, and eval data
# that you should download BEFORE terminating the pod.
#
# Does NOT include checkpoints (too large) — only metrics,
# configs, eval JSONs, and figures.
#
# Output:
#   results/m2/m2_results_bundle.tar.gz
#
# To download from RunPod:
#   Option 1 (RunPod web UI): Use the file manager to download
#   Option 2 (scp): scp root@<pod-ip>:/workspace/energy-aware-moe-router/results/m2/m2_results_bundle.tar.gz ./
#   Option 3 (rsync): rsync -avz root@<pod-ip>:/workspace/energy-aware-moe-router/results/m2/ ./results_m2/
#
# Usage:
#   bash scripts/runpod/06_package.sh
# ============================================================
set -euo pipefail

echo "============================================"
echo " Step 6: Package Results"
echo " $(date)"
echo "============================================"

BUNDLE="results/m2/m2_results_bundle.tar.gz"

# Collect everything except checkpoints (too large)
tar czf "$BUNDLE" \
    --exclude='*/checkpoints/*' \
    results/m2/ \
    figures/m2/ \
    configs/tiny_moe.yaml \
    logs/ \
    2>/dev/null || true

SIZE=$(du -h "$BUNDLE" | cut -f1)
echo ""
echo "Bundle created: $BUNDLE ($SIZE)"
echo ""
echo "Contents:"
tar tzf "$BUNDLE" | head -30
echo "... (truncated)"
echo ""

# ── Final summary ──
echo "============================================"
echo " M2 Summary"
echo "============================================"

python3 -c "
import json, os

combined = 'results/m2/all_evals.jsonl'
if not os.path.exists(combined):
    print('No evaluation data found.')
    exit()

print()
print(f'{\"Router\":>15s}  {\"Lambda\":>7s}  {\"PPL\":>8s}  {\"Joules\":>10s}')
print('=' * 45)

with open(combined) as f:
    for line in f:
        d = json.loads(line)
        lam = d.get('lambda_energy', 0.0)
        lam_str = f'{lam:.2f}' if d['router_type'] == 'energy_aware' else '-'
        print(f'{d[\"router_type\"]:>15s}  {lam_str:>7s}  {d[\"ppl\"]:>8.2f}  {d[\"total_joules\"]:>10.1f}')

print()
print('Key files to download:')
print('  results/m2/all_evals.jsonl       (raw data for Pareto plot)')
print('  figures/m2/pareto_lambda_sweep.*  (the main figure)')
print('  figures/m2/training_curves.png    (training dynamics)')
print()
"

echo ""
echo "Download the bundle, then you can safely Stop or Terminate the pod."
echo ""
echo "To STOP (keep disk, resume later):  RunPod UI → Stop Pod"
echo "To TERMINATE (delete everything):   RunPod UI → Terminate Pod"
echo ""
echo "============================================"
echo " Done! $(date)"
echo "============================================"
