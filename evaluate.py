#!/usr/bin/env python3
"""
evaluate.py — Evaluation harness producing a single JSON row per run.

Loads a trained checkpoint and emits four numbers:
  1. Validation perplexity (PPL)
  2. Total joules consumed during evaluation (from pynvml)
  3. Mean router entropy across MoE layers
  4. Load-balance loss at convergence

Usage:
    python evaluate.py \
        --checkpoint results/m2/switch/checkpoints/final.pt \
        --data-dir ./data/processed/c4_1pct \
        --output results/m2/switch/eval.json

The output JSON is designed to be concatenated across runs:
    cat results/m2/*/eval.json | python -m json.tool

For the Pareto plot, each run produces one (PPL, joules) point.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader import DataConfig, build_dataloader
from src.models.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    e_gpu: torch.Tensor | None = None,
    max_batches: int | None = None,
) -> dict:
    """Run evaluation and return metrics dict."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_balance_loss = 0.0
    total_energy_loss = 0.0
    total_entropy = 0.0
    num_moe_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, aux = model(input_ids, e_gpu=e_gpu)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens

            if aux.get("per_layer"):
                total_balance_loss += aux["total_balance_loss"].item()
                total_energy_loss += aux["total_energy_loss"].item()
                total_entropy += aux["mean_router_entropy"].item()
                num_moe_batches += 1

            if (i + 1) % 100 == 0:
                running_ppl = math.exp(min(total_loss / total_tokens, 20.0))
                log.info(f"  Eval batch {i+1}: running PPL = {running_ppl:.2f}")

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))

    return {
        "ppl": round(ppl, 4),
        "avg_loss": round(avg_loss, 6),
        "total_tokens": total_tokens,
        "balance_loss": round(total_balance_loss / max(num_moe_batches, 1), 6),
        "energy_loss": round(total_energy_loss / max(num_moe_batches, 1), 6),
        "mean_router_entropy": round(total_entropy / max(num_moe_batches, 1), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-dir", type=str, default="./data/processed/c4_1pct", help="Arrow dataset path")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit eval batches (for debugging)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    router_type = ckpt.get("router_type", "switch")

    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Router: {router_type}, Step: {ckpt.get('step', '?')}, Tokens: {ckpt.get('tokens', '?'):,}")

    # Device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Data
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer_name"], use_fast=True)

    data_cfg = DataConfig(
        arrow_dir=args.data_dir,
        tokenizer_name=cfg["data"]["tokenizer_name"],
        seq_len=cfg["data"]["seq_len"],
        batch_size=args.batch_size,
        num_workers=cfg["data"].get("num_workers", 4),
        seed=cfg["data"]["seed"],
    )
    eval_loader = build_dataloader(data_cfg)

    # Model
    model = build_model(cfg, tokenizer.vocab_size, router_type=router_type)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    log.info(f"Model loaded: {model.num_parameters:,} params")

    # Energy monitor for joules measurement
    energy_monitor = None
    e_gpu = None
    try:
        from src.utils.energy_monitor import EnergyMonitor
        energy_monitor = EnergyMonitor(
            device_indices=[0],
            sampling_hz=cfg.get("energy", {}).get("sampling_hz", 10),
        )
        energy_monitor.start()

        if router_type == "energy_aware":
            e_gpu_list = energy_monitor.latest_power_all_normalized()
            num_experts = cfg["moe"]["num_experts"]
            e_gpu = torch.tensor(e_gpu_list, device=device).repeat(
                math.ceil(num_experts / len(e_gpu_list))
            )[:num_experts]
    except Exception as e:
        log.warning(f"Energy monitor unavailable: {e}")

    # Evaluate
    log.info("Starting evaluation ...")
    t0 = time.time()
    metrics = evaluate(model, eval_loader, device, e_gpu, args.max_batches)
    wall_time = time.time() - t0

    # Energy
    total_joules = 0.0
    if energy_monitor is not None:
        summary = energy_monitor.stop()
        total_joules = sum(
            d["total_joules"] for d in summary.get("devices", {}).values()
        )

    # Assemble output
    result = {
        "router_type": router_type,
        "lambda_energy": cfg.get("energy", {}).get("lambda_energy", 0.0),
        "seed": cfg["training"]["seed"],
        "step": ckpt.get("step"),
        "tokens_trained": ckpt.get("tokens"),
        "ppl": metrics["ppl"],
        "total_joules": round(total_joules, 2),
        "balance_loss": metrics["balance_loss"],
        "mean_router_entropy": metrics["mean_router_entropy"],
        "wall_seconds": round(wall_time, 2),
        "eval_tokens": metrics["total_tokens"],
        "checkpoint": str(args.checkpoint),
    }

    # Output
    result_json = json.dumps(result, indent=2)
    log.info(f"\n{result_json}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(result_json + "\n")
        log.info(f"Saved to {out_path}")

    return result


if __name__ == "__main__":
    main()
