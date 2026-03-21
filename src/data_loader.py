"""
data_loader.py — Efficient data loading for C4 language modeling.

Supports two modes:
  1. **Arrow (local)**  — loads the pre-subsampled Arrow dataset from disk.
     Fast random access, shuffling, and slicing.  Use this for the 1% C4
     subsample during development and ablation runs.

  2. **Streaming (remote)** — streams directly from HuggingFace Hub.
     Zero disk footprint beyond the cache; ideal for full-scale runs on
     ephemeral cloud instances where you don't want to download 750 GB.

Both modes produce tokenized, fixed-length sequences suitable for causal
language modeling (next-token prediction).

Architecture notes
------------------
- Tokenizer: We default to the T5 SentencePiece tokenizer (same as the
  original Switch Transformer paper) so our perplexity numbers are directly
  comparable.  You can override this via ``--tokenizer``.
- Packing: Documents are concatenated with an <eos> separator and then
  chunked into ``seq_len``-length windows.  No padding is used, which
  maximises GPU utilisation.  This is standard practice for LM pre-training
  (GPT-2, T5, Switch Transformer all do this).
- The DataLoader is built on top of PyTorch's IterableDataset for the
  streaming path and a standard map-style Dataset for the Arrow path,
  so it integrates cleanly with DeepSpeed's data parallelism.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """All knobs for the data pipeline in one place."""

    # Data source — exactly one of these should be set.
    arrow_dir: Optional[str] = None          # path to save_to_disk() output
    hf_dataset: str = "allenai/c4"           # HuggingFace dataset name
    hf_config: str = "en.noclean"            # HuggingFace config name
    hf_split: str = "train"

    # Tokenizer
    tokenizer_name: str = "google-t5/t5-small"  # any HF tokenizer
    seq_len: int = 512                       # tokens per training example

    # DataLoader
    batch_size: int = 32
    num_workers: int = 4                     # 0 = main-process loading
    prefetch_factor: int = 2
    seed: int = 42

    # Streaming-specific
    streaming: bool = False
    streaming_buffer_size: int = 10_000      # shuffle buffer for streaming

    @property
    def use_arrow(self) -> bool:
        return self.arrow_dir is not None and not self.streaming


# ──────────────────────────────────────────────────────────────
# Tokenizer wrapper (lazy init, picklable)
# ──────────────────────────────────────────────────────────────

class _Tokenizer:
    """Thin wrapper so the tokenizer survives pickling across DataLoader workers."""

    def __init__(self, name: str):
        self.name = name
        self._tok = None

    @property
    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.name, use_fast=True)
        return self._tok

    def __call__(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    @property
    def eos_id(self) -> int:
        return self.tok.eos_token_id

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size

    def __getstate__(self):
        return {"name": self.name}

    def __setstate__(self, state):
        self.name = state["name"]
        self._tok = None


# ──────────────────────────────────────────────────────────────
# Document packing (concat + chunk)
# ──────────────────────────────────────────────────────────────

def _pack_tokens(
    token_iter: Iterator[list[int]],
    seq_len: int,
    eos_id: int,
) -> Iterator[list[int]]:
    """Concatenate documents with <eos> and yield fixed-length chunks.

    This is an online algorithm — it never materialises more than one
    chunk in memory, making it safe for the streaming path.
    """
    buf: list[int] = []
    for doc_ids in token_iter:
        buf.extend(doc_ids)
        buf.append(eos_id)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]
    # Drop the last partial chunk (standard for LM pre-training).


# ──────────────────────────────────────────────────────────────
# Arrow (map-style) Dataset
# ──────────────────────────────────────────────────────────────

class ArrowLMDataset(Dataset):
    """Map-style dataset backed by a pre-tokenized Arrow file on disk.

    On first use this class tokenizes and packs the raw text into
    fixed-length chunks, then caches the result so subsequent epochs
    load instantly.
    """

    def __init__(self, cfg: DataConfig):
        from datasets import load_from_disk

        self.cfg = cfg
        self.tokenizer = _Tokenizer(cfg.tokenizer_name)

        log.info(f"Loading Arrow dataset from {cfg.arrow_dir}")
        raw = load_from_disk(cfg.arrow_dir)
        log.info(f"  {len(raw):,} raw documents")

        # Tokenize + pack
        cache_path = Path(cfg.arrow_dir) / f".packed_seqlen{cfg.seq_len}.pt"
        if cache_path.exists():
            log.info(f"  Loading cached packed sequences from {cache_path}")
            self.data = torch.load(cache_path, weights_only=True)
        else:
            log.info("  Tokenizing and packing (first time — will be cached) …")
            token_iter = (self.tokenizer(ex["text"]) for ex in raw)
            chunks = list(
                _pack_tokens(token_iter, cfg.seq_len, self.tokenizer.eos_id)
            )
            self.data = torch.tensor(chunks, dtype=torch.long)
            torch.save(self.data, cache_path)
            log.info(f"  Cached {len(self.data):,} packed sequences to {cache_path}")

        log.info(f"  Final: {len(self.data):,} sequences of length {cfg.seq_len}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {
            "input_ids": tokens[:-1],       # [seq_len - 1]
            "labels": tokens[1:],           # [seq_len - 1]
        }


# ──────────────────────────────────────────────────────────────
# Streaming (iterable) Dataset
# ──────────────────────────────────────────────────────────────

class StreamingLMDataset(IterableDataset):
    """Iterable dataset that streams C4 from HuggingFace Hub.

    Handles multi-worker sharding automatically via
    ``torch.utils.data.get_worker_info()``.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.tokenizer = _Tokenizer(cfg.tokenizer_name)

    def _get_stream(self):
        from datasets import load_dataset

        ds = load_dataset(
            self.cfg.hf_dataset,
            self.cfg.hf_config,
            split=self.cfg.hf_split,
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(
            seed=self.cfg.seed,
            buffer_size=self.cfg.streaming_buffer_size,
        )
        return ds

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        stream = self._get_stream()

        # Shard across DataLoader workers
        if worker_info is not None:
            # Each worker skips examples that aren't "theirs"
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            stream = (
                ex
                for i, ex in enumerate(stream)
                if i % num_workers == worker_id
            )

        token_iter = (self.tokenizer(ex["text"]) for ex in stream)
        for chunk in _pack_tokens(
            token_iter, self.cfg.seq_len, self.tokenizer.eos_id
        ):
            tokens = torch.tensor(chunk, dtype=torch.long)
            yield {
                "input_ids": tokens[:-1],
                "labels": tokens[1:],
            }


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def build_dataset(cfg: DataConfig) -> Dataset | IterableDataset:
    """Factory: return the right dataset class based on config."""
    if cfg.use_arrow:
        return ArrowLMDataset(cfg)
    else:
        return StreamingLMDataset(cfg)


def build_dataloader(cfg: DataConfig) -> DataLoader:
    """Build a ready-to-iterate DataLoader."""
    ds = build_dataset(cfg)

    common_kwargs = dict(
        batch_size=cfg.batch_size,
        pin_memory=True,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    if isinstance(ds, IterableDataset):
        return DataLoader(ds, **common_kwargs)
    else:
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        return DataLoader(
            ds,
            shuffle=True,
            generator=g,
            drop_last=True,
            **common_kwargs,
        )


# ──────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the data loader.")
    parser.add_argument("--arrow-dir", type=str, default="./data/processed/c4_1pct")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-batches", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = DataConfig(
        arrow_dir=None if args.streaming else args.arrow_dir,
        streaming=args.streaming,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=0,  # simpler for a quick test
    )

    loader = build_dataloader(cfg)
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        print(
            f"Batch {i}: input_ids {batch['input_ids'].shape}, "
            f"labels {batch['labels'].shape}"
        )
    print("Smoke test passed.")
