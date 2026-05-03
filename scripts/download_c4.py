#!/usr/bin/env python3
"""
Download the C4 (en.noclean) dataset and create a 1% random subsample.

Usage
-----
# Full download + subsample (default):
    python scripts/download_c4.py

# Custom subsample ratio and seed:
    python scripts/download_c4.py --ratio 0.01 --seed 42 --cache-dir ./data/raw

# Skip the full download if you only want streaming (subsample only):
    python scripts/download_c4.py --streaming-subsample --ratio 0.01

Notes
-----
- The full C4 en.noclean split is ~750 GB on disk (305 GB compressed).
  Make sure you have sufficient storage before running without --streaming-subsample.
- The 1% subsample (~7.5 GB) is saved to data/processed/c4_1pct/ as an
  Arrow dataset that loads instantly for training.
- We use a deterministic hash-based sampling strategy so the subsample is
  reproducible across machines without needing to shuffle the full dataset
  into memory.
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic hash-based sampling
# ---------------------------------------------------------------------------

def _hash_sample(example: dict, ratio: float, seed: int) -> bool:
    """Return True if this example should be kept.

    We hash (seed || text) with MD5 and check whether the resulting
    integer modulo 10000 falls below ratio * 10000.  This is:
      - deterministic (same text + seed → same decision on any machine)
      - O(1) memory (no need to load the full dataset)
      - uniformly random to a good approximation for ratio >= 1e-4
    """
    text = example.get("text", "")
    digest = hashlib.md5(f"{seed}|{text[:512]}".encode()).hexdigest()
    return (int(digest, 16) % 10_000) < int(ratio * 10_000)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_full(cache_dir: str) -> "datasets.Dataset":
    """Download the full C4 en.noclean split to *cache_dir*."""
    from datasets import load_dataset

    log.info(
        "Downloading C4 en.noclean (this is ~305 GB compressed, ~750 GB on disk). "
        "Go get coffee."
    )
    ds = load_dataset(
        "allenai/c4",
        "en.noclean",
        split="train",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    log.info(f"Full dataset loaded: {len(ds):,} examples")
    return ds


def create_subsample_from_full(
    ds, ratio: float, seed: int, output_dir: str
) -> None:
    """Filter an in-memory / memory-mapped dataset to *ratio* and save."""
    log.info(f"Subsampling at ratio={ratio} with seed={seed} …")
    sub = ds.filter(
        lambda ex: _hash_sample(ex, ratio, seed),
        num_proc=os.cpu_count(),
        desc="Subsampling",
    )
    log.info(f"Subsample size: {len(sub):,} examples")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sub.save_to_disk(str(out))
    log.info(f"Saved subsample to {out}")


def create_subsample_streaming(
    ratio: float, seed: int, output_dir: str, max_examples: int = 0
) -> None:
    """Stream through C4 without downloading the full dataset first.

    This is much friendlier on disk (only the subsample is persisted) but
    slower because every example must be fetched over the network.

    WARNING: With ratio=0.01 and no max_examples, this scans ALL 7,168
    shards (~100+ hours). Use --fast mode or set --max-examples instead.
    """
    from datasets import load_dataset, Dataset

    log.info(
        f"Streaming C4 en.noclean and subsampling at ratio={ratio} "
        f"(seed={seed}) …"
    )
    stream = load_dataset(
        "allenai/c4",
        "en.noclean",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    kept: list[dict] = []
    seen = 0
    for example in stream:
        seen += 1
        if _hash_sample(example, ratio, seed):
            kept.append(example)
        if seen % 1_000_000 == 0:
            log.info(
                f"  … streamed {seen:,} examples, kept {len(kept):,}"
            )
        if max_examples and len(kept) >= max_examples:
            log.info(f"Reached max_examples={max_examples}, stopping.")
            break

    log.info(f"Finished streaming. Total seen: {seen:,}, kept: {len(kept):,}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(kept)
    ds.save_to_disk(str(out))
    log.info(f"Saved subsample to {out}")


def create_subsample_fast(
    output_dir: str, max_examples: int = 500_000
) -> None:
    """Take the first N examples sequentially from C4 streaming.

    This is MUCH faster than hash-based sampling because it reads shards
    sequentially and stops as soon as it has enough data. No need to scan
    all 7,168 shards.

    500K examples ≈ 100M tokens — more than enough for a 16M param model
    training on 50M tokens. Takes ~15-20 minutes vs 100+ hours for full scan.
    """
    from datasets import load_dataset, Dataset

    log.info(
        f"Fast sequential download: taking first {max_examples:,} examples from C4 en.noclean"
    )
    stream = load_dataset(
        "allenai/c4",
        "en.noclean",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    kept: list[dict] = []
    for i, example in enumerate(stream):
        if i >= max_examples:
            break
        kept.append(example)
        if (i + 1) % 50_000 == 0:
            log.info(f"  … collected {i + 1:,} / {max_examples:,} examples")

    log.info(f"Collected {len(kept):,} examples")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(kept)
    ds.save_to_disk(str(out))
    log.info(f"Saved to {out}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_subsample(output_dir: str, expected_ratio: float) -> None:
    """Quick sanity checks on the persisted subsample."""
    from datasets import load_from_disk

    ds = load_from_disk(output_dir)
    n = len(ds)
    log.info(f"Validation — loaded {n:,} examples from {output_dir}")

    # Check a few examples have non-empty text
    sample = ds.select(range(min(5, n)))
    for i, ex in enumerate(sample):
        text = ex.get("text", "")
        assert len(text) > 0, f"Example {i} has empty text!"
        log.info(f"  Example {i}: {len(text)} chars — {text[:80]!r}…")

    # Rough size check (C4 train has ~364M examples)
    c4_full_approx = 364_000_000
    expected = int(c4_full_approx * expected_ratio)
    lo, hi = int(expected * 0.8), int(expected * 1.2)
    if not (lo <= n <= hi):
        log.warning(
            f"Subsample size {n:,} is outside the expected range "
            f"[{lo:,}, {hi:,}] for ratio={expected_ratio}. "
            "This might be fine if you used --max-examples or --streaming-subsample."
        )
    else:
        log.info(f"Subsample size looks reasonable (expected ~{expected:,}).")

    log.info("Validation passed ✓")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download C4 en.noclean and create a reproducible subsample."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data/raw",
        help="Where to cache the full HuggingFace download (default: ./data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed/c4_1pct",
        help="Where to save the Arrow subsample (default: ./data/processed/c4_1pct)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.01,
        help="Fraction of examples to keep (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic hashing (default: 42)",
    )
    parser.add_argument(
        "--streaming-subsample",
        action="store_true",
        help="Stream C4 instead of downloading the full dataset first. "
        "Saves disk but is slower.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Stop after collecting this many subsample examples (0 = no limit). "
        "Useful for quick tests.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the post-download sanity checks.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast sequential download: take the first N examples without "
        "hash-sampling across all shards. Much faster (~15-20 min vs 100+ hrs). "
        "Use --max-examples to control how many (default: 500K).",
    )

    args = parser.parse_args()

    if args.fast:
        max_ex = args.max_examples if args.max_examples > 0 else 500_000
        create_subsample_fast(
            output_dir=args.output_dir,
            max_examples=max_ex,
        )
    elif args.streaming_subsample:
        create_subsample_streaming(
            ratio=args.ratio,
            seed=args.seed,
            output_dir=args.output_dir,
            max_examples=args.max_examples,
        )
    else:
        ds = download_full(args.cache_dir)
        create_subsample_from_full(
            ds,
            ratio=args.ratio,
            seed=args.seed,
            output_dir=args.output_dir,
        )

    if not args.skip_validation:
        validate_subsample(args.output_dir, args.ratio)


if __name__ == "__main__":
    main()
