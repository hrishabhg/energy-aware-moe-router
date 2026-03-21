# Energy-Aware Expert Routing in Mixture-of-Experts Models

Novel energy-aware routing for MoE models. The standard top-k gating function is augmented with a real-time energy penalty term `λ·e` derived from per-GPU power states, trained with a multi-objective loss that balances perplexity against energy consumption (Joules).

**Target venue:** Sustainable AI workshop at NeurIPS or SustaiNLP at EMNLP.

---

## Setup

### Prerequisites

Install [Miniforge](https://github.com/conda-forge/miniforge) (recommended) or Miniconda.

### Local development (Mac / CPU)

```bash
conda env create -f environment.yml
conda activate energy-moe
```

### GPU training (RunPod / Linux + NVIDIA)

```bash
conda env create -f environment-gpu.yml
conda activate energy-moe
```

This adds CUDA 12.4, DeepSpeed, and `pynvml` for GPU power monitoring.

### Prepare the data

```bash
# Quick test (~10K examples, ~2 min)
python scripts/download_c4.py \
    --streaming-subsample \
    --output-dir ./data/processed/c4_tiny \
    --ratio 0.01 --seed 42 --max-examples 10000

# Full 1% subsample (~3.6M examples, several hours streaming)
python scripts/download_c4.py \
    --streaming-subsample \
    --output-dir ./data/processed/c4_1pct \
    --ratio 0.01 --seed 42
```

### Verify the pipeline

```bash
python -m src.data_loader \
    --arrow-dir ./data/processed/c4_tiny \
    --batch-size 4 --seq-len 512 --num-batches 3
```

See [`data/README.md`](data/README.md) for more data options.

---

## Project Structure

```
energy-aware-moe-router/
├── data/
│   ├── raw/                        # HuggingFace cache (gitignored)
│   ├── processed/                  # Arrow subsamples (gitignored)
│   └── README.md
├── src/
│   ├── models/
│   │   └── energy_aware_moe.py     # (M3) Core contribution
│   ├── baselines/
│   │   ├── switch_transformer.py   # (M2) λ=0 baseline
│   │   └── random_router.py        # (M2) Lower bound
│   ├── utils/
│   │   └── energy_monitor.py       # pynvml GPU power polling
│   ├── data_loader.py              # Tokenization, packing, DataLoader
│   └── run_experiment.py           # Training loop with DeepSpeed
├── configs/                        # YAML experiment configs
├── scripts/
│   └── download_c4.py              # Data download + subsample
├── notebooks/
├── results/                        # (gitignored)
├── figures/
├── tables/
├── manuscript/
├── literature_review/
├── environment.yml                 # Conda env — local / Mac
├── environment-gpu.yml             # Conda env — RunPod / GPU
└── README.md
```

---

## Milestones

| Milestone | Weeks | Focus |
|-----------|-------|-------|
| M1 | 1–2 | Literature review & foundations |
| **M2** | **3–4** | **Environment, data pipeline, baselines** ← current |
| M3 | 5–6 | Core experiments & ablations |
| M4 | 7–8 | Evaluation & analysis |
| M5 | 9–10 | Manuscript & submission |

---

## Design Decisions

**Tokenizer:** T5 SentencePiece — matches the Switch Transformer paper for direct perplexity comparison.

**Document packing:** Concatenate with `<eos>`, chunk to fixed length, no padding. Standard for LM pre-training.

**Subsample:** Deterministic hash-based (`MD5(seed || text[:512]) mod 10000`). Reproducible across machines without coordinating shuffles over 364M examples.

**Energy monitoring:** `pynvml` polls per-GPU power at ~1 Hz, averaged over 1-second windows. Feeds the cost vector `[e₁, e₂, …, eₙ]` into the gating network.
