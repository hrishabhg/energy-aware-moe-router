#!/usr/bin/env python3
"""
train.py — Unified training loop for all routing strategies.

Usage:
    python train.py --config configs/tiny_moe.yaml --router switch
    python train.py --config configs/tiny_moe.yaml --router random
    python train.py --config configs/tiny_moe.yaml --router dense
    python train.py --config configs/tiny_moe.yaml --router energy_aware --lambda-energy 0.1

All four variants use the same training loop, same data, same evaluation.
The ONLY difference is the --router flag, which controls what sits in the
FFN slot of layers 2 and 4.

Outputs are saved to results/m2/<router_type>/:
    - checkpoints/       model checkpoints
    - metrics.jsonl      per-step metrics (loss, PPL, energy, etc.)
    - config.yaml        frozen config for reproducibility
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader import DataConfig, build_dataloader
from src.models.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CLI ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tiny MoE model.")
    parser.add_argument(
        "--config", type=str, default="configs/tiny_moe.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--router", type=str, default="switch",
        choices=["switch", "energy_aware", "random", "dense"],
        help="Routing strategy",
    )
    parser.add_argument(
        "--lambda-energy", type=float, default=None,
        help="Override lambda_energy from config (energy_aware only)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed from config",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--no-cuda", action="store_true",
        help="Force CPU training (for debugging)",
    )
    return parser.parse_args()


# ── Training step ──────────────────────────────────────────────

def train_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: dict,
    e_gpu: torch.Tensor | None,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """Single training step. Returns metrics dict."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=use_amp):
        # Forward pass
        logits, aux = model(input_ids, e_gpu=e_gpu)

        # Task loss: cross-entropy for next-token prediction
        # logits: (B, T, vocab_size), labels: (B, T)
        loss_task = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # Auxiliary losses from MoE layers
        alpha = cfg["moe"].get("alpha_balance", 0.01)
        beta = cfg["moe"].get("beta_energy", 0.0)

        balance_loss = aux.get("total_balance_loss", torch.tensor(0.0, device=device))
        energy_loss = aux.get("total_energy_loss", torch.tensor(0.0, device=device))

        # Total loss
        loss = loss_task + alpha * balance_loss + beta * energy_loss

    # Backward pass with gradient scaling (for mixed precision)
    scaler.scale(loss).backward()

    # Gradient clipping
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), cfg["training"].get("grad_clip", 1.0)
    )

    scaler.step(optimizer)
    scaler.update()

    # Perplexity: exp(cross-entropy loss)
    ppl = math.exp(min(loss_task.item(), 20.0))  # cap to avoid overflow

    metrics = {
        "loss": loss.item(),
        "loss_task": loss_task.item(),
        "ppl": ppl,
        "balance_loss": balance_loss.item() if torch.is_tensor(balance_loss) else balance_loss,
        "energy_loss": energy_loss.item() if torch.is_tensor(energy_loss) else energy_loss,
        "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
        "mean_router_entropy": aux.get("mean_router_entropy", torch.tensor(0.0)).item(),
    }

    # Per-layer expert load for logging
    if aux.get("per_layer"):
        for i, layer_aux in enumerate(aux["per_layer"]):
            load = layer_aux["expert_load"]
            metrics[f"expert_load_layer{i}"] = load.tolist()

    return metrics


# ── Learning rate schedule ─────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    """Linear warmup + cosine decay schedule."""
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    else:
        # Cosine decay to 10% of max_lr
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ── Main training loop ─────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override config with CLI args
    if args.lambda_energy is not None:
        cfg["energy"]["lambda_energy"] = args.lambda_energy
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
        cfg["data"]["seed"] = args.seed

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        use_amp = False
    else:
        device = torch.device("cuda")
        use_amp = cfg["training"].get("dtype", "float16") == "float16"

    log.info(f"Device: {device}, AMP: {use_amp}")
    log.info(f"Router: {args.router}, Lambda: {cfg['energy'].get('lambda_energy', 0.0)}")

    # ── Data ──
    data_cfg = DataConfig(
        arrow_dir=cfg["data"]["arrow_dir"],
        tokenizer_name=cfg["data"]["tokenizer_name"],
        seq_len=cfg["data"]["seq_len"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
        seed=seed,
    )
    train_loader = build_dataloader(data_cfg)

    # Get vocab size from tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.tokenizer_name, use_fast=True)
    vocab_size = tokenizer.vocab_size
    log.info(f"Vocab size: {vocab_size}")

    # ── Model ──
    model = build_model(cfg, vocab_size, router_type=args.router)
    model = model.to(device)
    log.info(f"Model parameters: {model.num_parameters:,} total, "
             f"{model.num_parameters_no_emb:,} non-embedding")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Energy monitor (if applicable) ──
    e_gpu = None
    energy_monitor = None
    if args.router == "energy_aware":
        from src.utils.energy_monitor import EnergyMonitor
        num_experts = cfg["moe"]["num_experts"]
        energy_monitor = EnergyMonitor(
            device_indices=list(range(min(num_experts, torch.cuda.device_count()))) if torch.cuda.is_available() else [0],
            sampling_hz=cfg["energy"].get("sampling_hz", 10),
            tdp_watts=None,  # TODO: auto-detect from pynvml
        )
        energy_monitor.start()

    # ── Output directory ──
    out_dir = Path(cfg["output"]["dir"]) / args.router
    if args.lambda_energy is not None and args.router == "energy_aware":
        out_dir = out_dir / f"lambda_{args.lambda_energy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Save frozen config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── W&B ──
    if args.wandb:
        import wandb
        wandb.init(
            project=cfg["logging"].get("wandb_project", "energy-aware-moe"),
            entity=cfg["logging"].get("wandb_entity"),
            name=f"{args.router}_seed{seed}",
            config=cfg,
        )

    # ── Training ──
    tokens_per_batch = cfg["data"]["batch_size"] * (cfg["data"]["seq_len"] - 1)
    max_tokens = cfg["training"]["max_tokens"]
    max_steps = max_tokens // tokens_per_batch
    warmup_tokens = cfg["training"].get("warmup_tokens", 1_000_000)
    warmup_steps = warmup_tokens // tokens_per_batch

    log_interval = cfg["logging"].get("log_interval", 100)
    save_interval = cfg["logging"].get("save_interval", 5000)

    log.info(f"Max tokens: {max_tokens:,}, Max steps: {max_steps:,}, "
             f"Warmup steps: {warmup_steps:,}")

    metrics_file = open(out_dir / "metrics.jsonl", "w")
    model.train()
    step = 0
    tokens_seen = 0
    t0 = time.time()

    while tokens_seen < max_tokens:
        for batch in train_loader:
            if tokens_seen >= max_tokens:
                break

            # Learning rate schedule
            lr = get_lr(step, warmup_steps, max_steps, cfg["training"]["lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Get energy vector if energy-aware
            if energy_monitor is not None:
                e_gpu_list = energy_monitor.latest_power_all_normalized()
                e_gpu = torch.tensor(e_gpu_list, device=device, dtype=torch.float32)
                # If fewer GPUs than experts, tile the readings
                num_experts = cfg["moe"]["num_experts"]
                if len(e_gpu_list) < num_experts:
                    e_gpu = e_gpu.repeat(math.ceil(num_experts / len(e_gpu_list)))[:num_experts]

            # Train step
            metrics = train_step(
                model, batch, optimizer, scaler, cfg, e_gpu, device, use_amp
            )
            metrics["step"] = step
            metrics["tokens"] = tokens_seen
            metrics["lr"] = lr
            metrics["wall_time"] = time.time() - t0

            # Log
            if step % log_interval == 0:
                log.info(
                    f"step {step:>6d} | tokens {tokens_seen:>10,} | "
                    f"loss {metrics['loss']:.4f} | ppl {metrics['ppl']:.2f} | "
                    f"lr {lr:.2e} | entropy {metrics['mean_router_entropy']:.3f}"
                )
                if args.wandb:
                    import wandb
                    wandb.log(metrics, step=step)

            # Save metrics
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()

            # Checkpoint
            if step > 0 and step % save_interval == 0:
                ckpt_path = ckpt_dir / f"step_{step}.pt"
                torch.save({
                    "step": step,
                    "tokens": tokens_seen,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "config": cfg,
                    "router_type": args.router,
                }, ckpt_path)
                log.info(f"Saved checkpoint: {ckpt_path}")

            step += 1
            tokens_seen += tokens_per_batch

    # ── Final checkpoint ──
    ckpt_path = ckpt_dir / "final.pt"
    torch.save({
        "step": step,
        "tokens": tokens_seen,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": cfg,
        "router_type": args.router,
    }, ckpt_path)
    log.info(f"Saved final checkpoint: {ckpt_path}")

    metrics_file.close()

    # ── Cleanup ──
    if energy_monitor is not None:
        summary = energy_monitor.stop()
        with open(out_dir / "energy_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Energy monitor overhead: {summary.get('sampling_overhead_pct', 0):.4f}%")

    if args.wandb:
        import wandb
        wandb.finish()

    wall_time = time.time() - t0
    log.info(f"Training complete. {step} steps, {tokens_seen:,} tokens, {wall_time:.1f}s")


if __name__ == "__main__":
    main()
