"""
transformer.py — Decoder-only Transformer building blocks (pure PyTorch).

This file contains the reusable components that are shared between
Dense and MoE transformer blocks:
  - MultiHeadSelfAttention
  - PositionwiseFFN (the standard dense feed-forward network)
  - TransformerBlock (attention + FFN, with pre-norm residuals)

The MoE-specific components (Expert, TopKGating, MoELayer) live in
moe.py. The full model assembly (TinyMoEModel) lives in model.py.

Design decisions:
  - Pre-LayerNorm (GPT-2 / GPT-3 style) rather than Post-LayerNorm.
    Pre-norm is more stable for training and is standard in modern LMs.
  - Causal mask is registered as a buffer so it moves to GPU automatically.
  - No bias in attention projections (following LLaMA / PaLM convention).
  - GELU activation in FFN (following GPT-2 / Switch Transformer).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Parameters
    ----------
    d_model : int
        Hidden dimension (256 in our testbed).
    n_heads : int
        Number of attention heads (4 in our testbed).
        d_k = d_v = d_model // n_heads = 64.
    dropout : float
        Dropout rate applied to attention weights.
    max_seq_len : int
        Maximum sequence length, used to pre-compute the causal mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 64 in our testbed

        # Combined QKV projection — more efficient than 3 separate projections
        # because it's a single matmul on the GPU.
        # Shape: (d_model) -> (3 * d_model) then split into Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection: concatenated heads back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Pre-compute causal mask: upper triangle is -inf, lower triangle + diagonal is 0
        # Shape: (1, 1, max_seq_len, max_seq_len) — broadcasts over batch and heads
        causal_mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, d_model)

        Returns
        -------
        Tensor of shape (batch, seq_len, d_model)
        """
        B, T, C = x.shape

        # Project to Q, K, V and split
        qkv = self.qkv_proj(x)                           # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B, n_heads, T, d_k)
        q, k, v = qkv.unbind(0)                          # each: (B, n_heads, T, d_k)

        # Scaled dot-product attention
        # attn_weights[i,j] = softmax(q_i · k_j / sqrt(d_k) + causal_mask[i,j])
        scale = math.sqrt(self.d_k)
        attn_weights = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)
        attn_weights = attn_weights + self.causal_mask[:, :, :T, :T]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = attn_weights @ v                    # (B, n_heads, T, d_k)

        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.out_proj(attn_output))


class PositionwiseFFN(nn.Module):
    """Standard 2-layer feed-forward network used in dense transformer blocks.

    Architecture: d_model -> d_ff -> GELU -> d_ff -> d_model -> dropout

    This is the component that gets *replaced* by a MoE layer in MoE blocks.
    Each expert in the MoE layer has the same architecture as this FFN.

    Parameters
    ----------
    d_model : int
        Input/output dimension (256).
    d_ff : int
        Hidden dimension (1024 = 4x expansion).
    dropout : float
        Dropout on the output.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)"""
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """A single transformer decoder block with pre-norm residuals.

    Structure (pre-LayerNorm):
        x -> LayerNorm -> MultiHeadAttn -> + residual
          -> LayerNorm -> FFN (or MoE)  -> + residual

    The ffn parameter accepts either a PositionwiseFFN (for dense blocks)
    or a MoELayer (for MoE blocks). This is how we swap in the MoE layer
    at positions specified by moe_layers in the config.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    ffn : nn.Module
        The feed-forward component. Either PositionwiseFFN or MoELayer.
    dropout : float
        Dropout rate for residual connections.
    max_seq_len : int
        Maximum sequence length for the causal mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn: nn.Module,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = ffn  # PositionwiseFFN for dense, MoELayer for MoE blocks

    def forward(
        self,
        x: torch.Tensor,
        e_gpu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)
        e_gpu : (num_experts,), optional
            Energy cost vector, passed through to MoE layers.
            Ignored by dense FFN blocks.

        Returns
        -------
        output : (batch, seq_len, d_model)
        aux : dict
            Auxiliary outputs from the FFN. Empty dict for dense blocks.
            For MoE blocks, contains 'balance_loss', 'router_entropy', etc.
        """
        # Attention sub-block (pre-norm)
        x = x + self.attn(self.ln1(x))

        # FFN sub-block (pre-norm)
        ffn_input = self.ln2(x)

        # MoELayer.forward(x, e_gpu) returns (output, aux_dict)
        # PositionwiseFFN.forward(x) returns just output
        # We use duck typing: if the FFN accepts e_gpu, pass it.
        from .moe import MoELayer
        if isinstance(self.ffn, MoELayer):
            ffn_output, aux = self.ffn(ffn_input, e_gpu=e_gpu)
        else:
            ffn_output = self.ffn(ffn_input)
            aux = {}

        x = x + ffn_output
        return x, aux
