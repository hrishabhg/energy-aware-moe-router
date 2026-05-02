"""
model.py — Full TinyMoE model assembly.

Combines the building blocks from transformer.py and moe.py into a
complete decoder-only language model with configurable routing.

The --router flag in train.py controls which variant is built:

    --router switch       → MoE layers with learned top-k gating (lambda=0)
    --router energy_aware → MoE layers with energy penalty (lambda>0)
    --router random       → MoE layers with uniform random routing
    --router dense        → No MoE layers at all (standard transformer)

All four variants share the same TransformerBlock + attention code.
The only difference is what sits in the FFN slot of layers 2 and 4.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerBlock, PositionwiseFFN
from .moe import MoELayer


class RandomGatingMoELayer(MoELayer):
    """MoE layer with uniform random routing (baseline 2).

    Overrides the gating forward pass to ignore the learned W_g and
    instead assign each token to a random expert. This establishes a
    lower bound: if Switch doesn't beat Random on PPL, the MoE
    container has a bug.
    """

    def forward(
        self,
        x: torch.Tensor,
        e_gpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        num_tokens = x_flat.shape[0]

        # Random expert assignment — uniform over [0, num_experts)
        top_k_indices = torch.randint(
            0, self.num_experts, (num_tokens, self.top_k),
            device=x.device,
        )
        # Equal weights (for k=1 this is all 1.0)
        top_k_weights = torch.ones(
            num_tokens, self.top_k, device=x.device, dtype=x.dtype
        ) / self.top_k

        # Compute approximate load stats for logging (no learned gate)
        expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        expert_counts.scatter_add_(
            0, top_k_indices[:, 0],
            torch.ones(num_tokens, device=x.device, dtype=x.dtype),
        )
        f_i = expert_counts / num_tokens

        aux = {
            "balance_loss": torch.tensor(0.0, device=x.device),
            "energy_loss": torch.tensor(0.0, device=x.device),
            "router_entropy": torch.tensor(float(torch.log(
                torch.tensor(float(self.num_experts))
            )), device=x.device),
            "expert_load": f_i.detach(),
        }

        # Dispatch (reuse parent's dispatch logic by building output manually)
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            mask = (top_k_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            expert_input = x_flat[mask]
            expert_output = self.experts[expert_idx](expert_input)
            # For random routing with equal weights, each expert gets 1/k weight
            weight = 1.0 / self.top_k
            output[mask] += weight * expert_output

        output = output.reshape(B, T, D)
        return output, aux


class TinyMoEModel(nn.Module):
    """Complete decoder-only language model with configurable routing.

    Architecture (from configs/tiny_moe.yaml):

        Token Embedding + Positional Embedding
            → Block 0 (Dense)
            → Block 1 (Dense)
            → Block 2 (MoE or Dense depending on --router)
            → Block 3 (Dense)
            → Block 4 (MoE or Dense depending on --router)
            → Block 5 (Dense)
        → Final LayerNorm
        → LM Head (weight-tied with embedding)

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (32128 for T5 tokenizer).
    d_model : int
        Hidden dimension (256).
    n_heads : int
        Number of attention heads (4).
    n_layers : int
        Total number of transformer blocks (6).
    d_ff : int
        FFN hidden dimension per expert (1024).
    dropout : float
        Dropout rate (0.1).
    max_seq_len : int
        Maximum sequence length (512).
    moe_layers : list[int]
        Which layer indices get MoE layers (default: [2, 4]).
    num_experts : int
        Number of experts per MoE layer (4).
    top_k : int
        Number of experts per token (1).
    router_type : str
        "switch" | "energy_aware" | "random" | "dense"
    lambda_energy : float
        Energy penalty coefficient (only used if router_type="energy_aware").
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        moe_layers: list[int] | None = None,
        num_experts: int = 4,
        top_k: int = 1,
        router_type: str = "switch",
        lambda_energy: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.router_type = router_type
        moe_layers = moe_layers or [2, 4]

        # ── Embeddings ──
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # ── Transformer blocks ──
        self.blocks = nn.ModuleList()
        for layer_idx in range(n_layers):
            if layer_idx in moe_layers and router_type != "dense":
                # MoE block — FFN is replaced by MoE layer
                if router_type == "random":
                    ffn = RandomGatingMoELayer(
                        d_model=d_model,
                        d_ff=d_ff,
                        num_experts=num_experts,
                        top_k=top_k,
                        dropout=dropout,
                        lambda_energy=0.0,
                    )
                else:
                    # "switch" (lambda=0) or "energy_aware" (lambda>0)
                    ffn = MoELayer(
                        d_model=d_model,
                        d_ff=d_ff,
                        num_experts=num_experts,
                        top_k=top_k,
                        dropout=dropout,
                        lambda_energy=lambda_energy if router_type == "energy_aware" else 0.0,
                    )
            else:
                # Dense block — standard FFN
                ffn = PositionwiseFFN(d_model, d_ff, dropout)

            self.blocks.append(
                TransformerBlock(d_model, n_heads, ffn, dropout, max_seq_len)
            )

        # ── Output head ──
        self.ln_f = nn.LayerNorm(d_model)

        # LM head shares weights with token embedding (weight tying).
        # This is standard practice (GPT-2, T5, LLaMA) and reduces
        # parameter count by ~8M in our testbed.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Xavier uniform for linear layers, normal for embeddings."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        e_gpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        input_ids : Tensor, shape (batch, seq_len)
            Token indices.
        e_gpu : Tensor, shape (num_experts,), optional
            Per-expert energy costs from EnergyMonitor.

        Returns
        -------
        logits : Tensor, shape (batch, seq_len, vocab_size)
            Next-token prediction logits.
        aux : dict
            Aggregated auxiliary losses and metrics from all MoE layers:
            - 'total_balance_loss': sum of balance losses across MoE layers
            - 'total_energy_loss': sum of energy losses across MoE layers
            - 'mean_router_entropy': mean router entropy across MoE layers
            - 'per_layer': list of per-layer aux dicts
        """
        B, T = input_ids.shape

        # ── Embeddings ──
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # ── Forward through all blocks ──
        all_aux = []
        for block in self.blocks:
            # TransformerBlock.forward() accepts e_gpu and passes it
            # through to MoE layers internally. Dense blocks ignore it.
            x, aux = block(x, e_gpu=e_gpu)
            if aux:
                all_aux.append(aux)

        # ── Output ──
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # ── Aggregate auxiliary losses ──
        aggregated = self._aggregate_aux(all_aux)
        return logits, aggregated

    def _aggregate_aux(self, all_aux: list[dict]) -> dict:
        """Combine auxiliary outputs from all MoE layers."""
        if not all_aux:
            return {
                "total_balance_loss": torch.tensor(0.0),
                "total_energy_loss": torch.tensor(0.0),
                "mean_router_entropy": torch.tensor(0.0),
                "per_layer": [],
            }

        device = all_aux[0]["balance_loss"].device
        return {
            "total_balance_loss": sum(a["balance_loss"] for a in all_aux),
            "total_energy_loss": sum(a["energy_loss"] for a in all_aux),
            "mean_router_entropy": sum(a["router_entropy"] for a in all_aux) / len(all_aux),
            "per_layer": all_aux,
        }

    def set_lambda_energy(self, lambda_energy: float) -> None:
        """Update energy penalty on all MoE layers at runtime."""
        for block in self.blocks:
            if isinstance(block.ffn, MoELayer):
                block.ffn.set_lambda_energy(lambda_energy)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_parameters_no_emb(self) -> int:
        """Trainable parameters excluding embeddings (for model-size comparison)."""
        emb_params = self.token_emb.weight.numel() + self.pos_emb.weight.numel()
        return self.num_parameters - emb_params


def build_model(cfg: dict, vocab_size: int, router_type: str = "switch") -> TinyMoEModel:
    """Factory function: build a TinyMoEModel from a config dict.

    Parameters
    ----------
    cfg : dict
        Loaded from configs/tiny_moe.yaml.
    vocab_size : int
        From the tokenizer.
    router_type : str
        "switch" | "energy_aware" | "random" | "dense"

    Returns
    -------
    TinyMoEModel ready for training.
    """
    model_cfg = cfg["model"]
    moe_cfg = cfg["moe"]
    energy_cfg = cfg.get("energy", {})

    return TinyMoEModel(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg.get("dropout", 0.1),
        max_seq_len=model_cfg.get("max_seq_len", 512),
        moe_layers=moe_cfg.get("moe_layers", [2, 4]),
        num_experts=moe_cfg.get("num_experts", 4),
        top_k=moe_cfg.get("top_k", 1),
        router_type=router_type,
        lambda_energy=energy_cfg.get("lambda_energy", 0.0),
    )
