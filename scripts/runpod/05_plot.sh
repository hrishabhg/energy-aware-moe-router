#!/bin/bash
# ============================================================
# 05_plot.sh — Generate the Pareto frontier plot
# ============================================================
# Reads all_evals.jsonl and produces:
#   figures/m2/pareto_lambda_sweep.png
#   figures/m2/pareto_lambda_sweep.pdf   (for LaTeX)
#   figures/m2/training_curves.png
#
# Usage:
#   bash scripts/runpod/05_plot.sh 2>&1 | tee logs/05_plot.log
# ============================================================
set -euo pipefail

echo "============================================"
echo " Step 5: Generate Plots"
echo " $(date)"
echo "============================================"

mkdir -p figures/m2

python3 << 'PYTHON_SCRIPT'
import json
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless server
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Load evaluation data ──
evals = []
with open('results/m2/all_evals.jsonl') as f:
    for line in f:
        evals.append(json.loads(line))

if not evals:
    print("ERROR: No evaluation data found in results/m2/all_evals.jsonl")
    exit(1)

print(f"Loaded {len(evals)} evaluation results.")

# ============================================================
# Plot 1: Pareto frontier (PPL vs Joules)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Separate baselines from energy-aware sweep
baselines = [e for e in evals if e['router_type'] != 'energy_aware']
sweep = [e for e in evals if e['router_type'] == 'energy_aware']

# Sort sweep by lambda
sweep.sort(key=lambda e: e.get('lambda_energy', 0))

# Plot baselines as distinct markers
marker_map = {'switch': 's', 'random': 'D', 'dense': '^'}
color_map = {'switch': '#2196F3', 'random': '#FF9800', 'dense': '#9C27B0'}
label_map = {'switch': 'Switch (λ=0)', 'random': 'Random', 'dense': 'Dense'}

for e in baselines:
    ax.scatter(
        e['total_joules'], e['ppl'],
        marker=marker_map.get(e['router_type'], 'o'),
        color=color_map.get(e['router_type'], 'gray'),
        s=150, zorder=5, edgecolors='black', linewidth=1.2,
        label=label_map.get(e['router_type'], e['router_type']),
    )

# Plot energy-aware sweep as connected line
if sweep:
    joules = [e['total_joules'] for e in sweep]
    ppls = [e['ppl'] for e in sweep]
    lambdas = [e.get('lambda_energy', 0) for e in sweep]

    ax.plot(joules, ppls, 'o-', color='#4CAF50', markersize=8,
            linewidth=2, label='Energy-Aware', zorder=4,
            markeredgecolor='black', markeredgewidth=0.8)

    # Annotate lambda values
    for j, p, lam in zip(joules, ppls, lambdas):
        ax.annotate(f'λ={lam}', (j, p),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=8, color='#333333')

ax.set_xlabel('Total Energy (Joules)', fontsize=13)
ax.set_ylabel('Perplexity (PPL)', fontsize=13)
ax.set_title('Energy-Quality Pareto Frontier (M2 Testbed)', fontsize=15)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

# Log scale on x-axis if range is large
if baselines or sweep:
    all_joules = [e['total_joules'] for e in evals if e['total_joules'] > 0]
    if all_joules and max(all_joules) / max(min(all_joules), 1) > 10:
        ax.set_xscale('log')

plt.tight_layout()
fig.savefig('figures/m2/pareto_lambda_sweep.png', dpi=200, bbox_inches='tight')
fig.savefig('figures/m2/pareto_lambda_sweep.pdf', bbox_inches='tight')
print("Saved: figures/m2/pareto_lambda_sweep.png")
print("Saved: figures/m2/pareto_lambda_sweep.pdf")
plt.close()

# ============================================================
# Plot 2: Training curves (loss over steps)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Load training metrics
router_colors = {
    'switch': '#2196F3', 'random': '#FF9800', 'dense': '#9C27B0',
}

# Panel 1: Task loss
# Panel 2: PPL
# Panel 3: Router entropy

for router in ['switch', 'random', 'dense']:
    path = f'results/m2/{router}/metrics.jsonl'
    if not os.path.exists(path):
        continue
    with open(path) as f:
        metrics = [json.loads(line) for line in f]
    steps = [m['step'] for m in metrics]
    color = router_colors.get(router, 'gray')
    label = router.capitalize()

    axes[0].plot(steps, [m['loss_task'] for m in metrics],
                 color=color, alpha=0.7, label=label)
    axes[1].plot(steps, [min(m['ppl'], 1000) for m in metrics],
                 color=color, alpha=0.7, label=label)
    if router != 'dense':
        axes[2].plot(steps, [m.get('mean_router_entropy', 0) for m in metrics],
                     color=color, alpha=0.7, label=label)

# Add a couple energy-aware runs for comparison
for lam in ['0.1', '0.5']:
    path = f'results/m2/energy_aware/lambda_{lam}/metrics.jsonl'
    if not os.path.exists(path):
        continue
    with open(path) as f:
        metrics = [json.loads(line) for line in f]
    steps = [m['step'] for m in metrics]
    label = f'EA λ={lam}'

    axes[0].plot(steps, [m['loss_task'] for m in metrics],
                 color='#4CAF50', alpha=0.5, linestyle='--', label=label)
    axes[1].plot(steps, [min(m['ppl'], 1000) for m in metrics],
                 color='#4CAF50', alpha=0.5, linestyle='--', label=label)
    axes[2].plot(steps, [m.get('mean_router_entropy', 0) for m in metrics],
                 color='#4CAF50', alpha=0.5, linestyle='--', label=label)

axes[0].set_title('Task Loss')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Perplexity')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('PPL')
axes[1].set_yscale('log')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].set_title('Router Entropy')
axes[2].set_xlabel('Step')
axes[2].set_ylabel('Entropy (nats)')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('figures/m2/training_curves.png', dpi=200, bbox_inches='tight')
print("Saved: figures/m2/training_curves.png")
plt.close()

print("\nAll plots generated.")
PYTHON_SCRIPT

echo ""
echo "============================================"
echo " Plots complete. Run 06_package.sh to bundle results."
echo " $(date)"
echo "============================================"
