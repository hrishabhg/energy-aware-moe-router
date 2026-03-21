# Data Directory

## Dataset: C4 (Colossal Clean Crawled Corpus) — `en.noclean`

The primary training data for this project. C4 is a ~750 GB cleaned version of Common Crawl, used as the standard benchmark in Switch Transformer and related MoE papers.

**Source:** [allenai/c4 on HuggingFace](https://huggingface.co/datasets/allenai/c4)

---

## Directory Layout

```
data/
├── raw/                  # HuggingFace cache (full C4 download)
│   └── (auto-populated by datasets library)
├── processed/
│   └── c4_1pct/          # 1% deterministic subsample (Arrow format)
│       ├── dataset_info.json
│       ├── state.json
│       └── data-*.arrow
└── README.md             # this file
```

---

## How to Prepare the Data

### Option A: Full download → subsample (recommended if you have disk space)

This downloads the entire C4 en.noclean split (~305 GB compressed, ~750 GB uncompressed) and then creates a deterministic 1% subsample.

```bash
cd energy-aware-moe-router
conda activate energy-moe

python scripts/download_c4.py \
    --cache-dir ./data/raw \
    --output-dir ./data/processed/c4_1pct \
    --ratio 0.01 \
    --seed 42
```

**Estimated time:** 2–6 hours depending on bandwidth.
**Disk required:** ~800 GB for full cache + ~7.5 GB for subsample.

### Option B: Streaming subsample (no full download)

Streams C4 from HuggingFace Hub and only persists the 1% subsample. Much less disk, but slower since every example is fetched over the network.

```bash
python scripts/download_c4.py \
    --streaming-subsample \
    --output-dir ./data/processed/c4_1pct \
    --ratio 0.01 \
    --seed 42
```

**Estimated time:** 8–24 hours (network-bound).
**Disk required:** ~7.5 GB for subsample only.

### Option C: Quick test subset

For debugging the pipeline before committing to a long download:

```bash
python scripts/download_c4.py \
    --streaming-subsample \
    --output-dir ./data/processed/c4_tiny \
    --ratio 0.01 \
    --seed 42 \
    --max-examples 10000
```

This grabs ~10K examples in a few minutes.

---

## Subsample Reproducibility

The subsampling uses a deterministic hash: `MD5(seed || text[:512]) mod 10000`. With `seed=42` and `ratio=0.01`, any machine will produce the exact same subset. This means you can share experiment results without sharing the data itself.

---

## Smoke-Testing the Data Loader

Once you have a subsample (even the tiny one), verify the full pipeline:

```bash
python -m src.data_loader \
    --arrow-dir ./data/processed/c4_1pct \
    --batch-size 4 \
    --seq-len 512 \
    --num-batches 3
```

Expected output:
```
Batch 0: input_ids torch.Size([4, 511]), labels torch.Size([4, 511])
Batch 1: input_ids torch.Size([4, 511]), labels torch.Size([4, 511])
Batch 2: input_ids torch.Size([4, 511]), labels torch.Size([4, 511])
Smoke test passed.
```

---

## Optional Validation Sets

These are not required for M2 but will be used in later milestones:

| Dataset       | Purpose                                  | Size   |
|---------------|------------------------------------------|--------|
| WikiText-103  | Fast prototyping, debugging              | ~500MB |
| GLUE          | Downstream task evaluation (M4)          | ~1 GB  |
| The Pile      | Cross-distribution robustness check      | ~825GB |
