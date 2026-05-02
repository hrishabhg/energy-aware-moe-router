"""
moe.py — Mixture-of-Experts layer with energy-aware routing (pure PyTorch).

This file contains the three core MoE components:
  - Expert         : a single expert (same architecture as PositionwiseFFN)
  - TopKGating     : the learned router that decides which expert processes each token
  - MoELayer       : the full MoE layer that combines gating + experts + auxiliary losses

=============================================================================
WALKTHROUGH — Read this carefully. This is the core of your research.
=============================================================================

The standard MoE gating (Shazeer 2017, Switch Transformer) works like this:

    1. Each token x (shape: d_model) is projected through a learned weight
       matrix W_g (shape: d_model x num_experts) to produce raw logits:

           logits = x @ W_g                        # shape: (num_experts,)

    2. We take the top-k experts with the highest logits:

           top_k_indices = TopK(logits, k)

    3. The gating weights are the softmax over ONLY the selected top-k logits
       (not all experts — this is important for sparsity):

           gate_weights = softmax(logits[top_k_indices])

    4. The output is the weighted sum of the selected experts' outputs:

           output = sum(gate_weights[i] * Expert_i(x) for i in top_k_indices)

YOUR CONTRIBUTION — Energy-Aware Routing:

    You add a penalty term BEFORE the top-k selection:

        logits = x @ W_g - lambda * e_gpu

    where e_gpu is a vector of per-expert energy costs (one per expert),
    read from pynvml in real time. This biases the router AWAY from
    experts on high-power GPUs and TOWARD experts on low-power GPUs.

    Key insight: lambda is a hyperparameter that controls the trade-off.
    - lambda = 0   → standard routing (no energy awareness)
    - lambda > 0   → routing prefers low-power experts
    - lambda too large → all tokens go to the lowest-power expert (collapse)

    The energy cost vector e_gpu is treated as a NON-DIFFERENTIABLE constant
    in the forward pass. Gradients do NOT flow through the energy measurement.
    The model learns W_g around a fixed energy landscape.

AUXILIARY LOSSES:

    L = L_task + alpha * L_balance + beta * L_energy

    L_task:    cross-entropy for next-token prediction (standard LM loss)

    L_balance: load-balancing loss to prevent expert collapse.
               Without this, the router often sends all tokens to 1-2 experts
               and the rest starve. We use the simplified Switch Transformer
               formulation:

               L_balance = num_experts * sum(f_i * p_i)

               where f_i = fraction of tokens routed to expert i (top-1 count)
                     p_i = mean router probability for expert i (before top-k)

               This is minimized when all experts get equal load (f_i = 1/E)
               and equal probability (p_i = 1/E).

    L_energy:  mean energy cost of the routed experts. This is your
               training-time pressure — it teaches W_g to prefer routing
               patterns that are energy-efficient. beta = 0 disables it
               (used for baselines).

               L_energy = mean(e_gpu[selected_expert_indices])

=============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single expert — identical architecture to PositionwiseFFN.

    Each expert is an independent 2-layer FFN:
        d_model -> d_ff -> GELU -> d_model

    In our testbed: 256 -> 1024 -> 256 (each expert has ~524K params).

    We keep this as a separate class (rather than reusing PositionwiseFFN)
    so it's explicit in the code that experts are the atomic unit of the
    MoE layer. When you read a stack trace or profile, you'll see "Expert"
    not "PositionwiseFFN" and immediately know it's inside the MoE.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(*, d_model) -> (*, d_model)"""
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TopKGating(nn.Module):
    """Learned top-k gating network with optional energy penalty.

    This is THE core module of your research. Every line matters.

    Forward pass:
        1. Project input tokens to expert logits:    logits = x @ W_g
        2. (Energy-aware) Subtract energy penalty:   logits -= lambda * e_gpu
        3. Select top-k experts per token:           indices = TopK(logits, k)
        4. Compute gate weights via softmax:         weights = softmax(selected_logits)
        5. Compute auxiliary losses:                 L_balance, L_energy

    Parameters
    ----------
    d_model : int
        Token embedding dimension (256).
    num_experts : int
        Number of experts to route across (4).
    top_k : int
        Number of experts activated per token (1 for Switch-style).
    lambda_energy : float
        Energy penalty coefficient. 0.0 = standard routing.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 1,
        lambda_energy: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.lambda_energy = lambda_energy

        # ── The gating weight matrix W_g ──
        # This is the ONLY learned parameter in the router.
        # Shape: (d_model, num_experts) = (256, 4)
        # Each column is a "prototype" for one expert.
        # The dot product x @ W_g measures how much token x
        # "wants" to go to each expert.
        self.W_g = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        e_gpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Parameters
        ----------
        x : Tensor, shape (batch * seq_len, d_model)
            Flattened token embeddings. We flatten batch and sequence
            dimensions because routing is per-token, not per-sequence.
        e_gpu : Tensor, shape (num_experts,), optional
            Per-expert energy cost vector from pynvml. Normalized to [0, 1].
            If None, no energy penalty is applied (equivalent to lambda=0).

        Returns
        -------
        top_k_indices : Tensor, shape (num_tokens, top_k)
            Which expert(s) each token is routed to.
        top_k_weights : Tensor, shape (num_tokens, top_k)
            The gating weight for each selected expert (sums to 1 per token).
        aux : dict containing:
            - 'balance_loss': scalar, the load-balance auxiliary loss
            - 'energy_loss': scalar, mean energy cost of selected experts
            - 'router_entropy': scalar, mean entropy of the full router distribution
            - 'expert_load': Tensor (num_experts,), fraction of tokens per expert
        """
        num_tokens = x.shape[0]  # batch * seq_len

        # ────────────────────────────────────────────────────────────
        # Step 1: Compute raw router logits
        # ────────────────────────────────────────────────────────────
        # logits[i, j] = how much token i wants to go to expert j
        # This is a learned linear projection — W_g is updated by backprop.
        logits = self.W_g(x)  # (num_tokens, num_experts)

        # ────────────────────────────────────────────────────────────
        # Step 2: Apply energy penalty (YOUR CONTRIBUTION)
        # ────────────────────────────────────────────────────────────
        # This is the key modification: subtract lambda * e_gpu from logits
        # BEFORE the top-k selection. This biases the router away from
        # high-energy experts.
        #
        # e_gpu is treated as a CONSTANT — no gradients flow through it.
        # The model learns W_g to route well given the energy landscape,
        # but the energy landscape itself is a physical measurement, not
        # a learnable parameter.
        #
        # Mathematically:
        #   logits_i = x^T @ w_i - lambda * e_i
        #
        # where e_i is the normalized power draw of the GPU hosting expert i.
        # Higher e_i → lower logits_i → less likely to be selected by top-k.
        if e_gpu is not None and self.lambda_energy > 0:
            # e_gpu shape: (num_experts,) — broadcasts over num_tokens
            # .detach() is belt-and-suspenders: e_gpu should already be
            # a non-leaf tensor from pynvml, but this makes intent explicit.
            logits = logits - self.lambda_energy * e_gpu.detach()

        # ────────────────────────────────────────────────────────────
        # Step 3: Compute router probabilities (BEFORE top-k selection)
        # ────────────────────────────────────────────────────────────
        # These probabilities are used for the balance loss and for
        # computing router entropy. They reflect the router's full
        # distribution over all experts, not just the selected ones.
        router_probs = F.softmax(logits, dim=-1)  # (num_tokens, num_experts)

        # ────────────────────────────────────────────────────────────
        # Step 4: Top-k expert selection
        # ────────────────────────────────────────────────────────────
        # For k=1 (Switch Transformer style): each token goes to exactly
        # one expert. This is what makes MoE "sparse" — only 1/E of the
        # experts are activated per token.
        #
        # For k=2: each token is processed by 2 experts and the outputs
        # are linearly combined. More compute but potentially better quality.
        top_k_logits, top_k_indices = torch.topk(
            logits, self.top_k, dim=-1
        )  # both: (num_tokens, top_k)

        # ────────────────────────────────────────────────────────────
        # Step 5: Compute gate weights from selected logits
        # ────────────────────────────────────────────────────────────
        # Softmax over ONLY the top-k logits, not all experts.
        # For k=1 this is always [1.0] (single expert gets full weight).
        # For k=2 this distributes weight between the two selected experts.
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (num_tokens, top_k)

        # ────────────────────────────────────────────────────────────
        # Step 6: Compute auxiliary losses
        # ────────────────────────────────────────────────────────────

        # --- Load-balance loss (Switch Transformer formulation) ---
        #
        # f_i = fraction of tokens assigned to expert i (from hard top-1 routing)
        #       This is a discrete count — not differentiable by itself.
        #
        # p_i = mean of router_probs[:, i] across all tokens
        #       This IS differentiable — gradients flow through softmax(logits).
        #
        # L_balance = num_experts * sum(f_i * p_i)
        #
        # Why this works: if expert i gets too many tokens (high f_i),
        # and the router is confident about sending them there (high p_i),
        # the loss is high. To reduce L_balance, backprop pushes p_i down
        # for overloaded experts, which redistributes tokens in the next step.
        #
        # The num_experts multiplier normalizes the loss so it's ~1.0
        # when load is perfectly balanced.

        # f_i: count how many tokens have each expert as their top-1 choice
        # Use the FIRST column of top_k_indices (the top-1 expert)
        top1_indices = top_k_indices[:, 0]  # (num_tokens,)
        expert_counts = torch.zeros(
            self.num_experts, device=x.device, dtype=x.dtype
        )
        expert_counts.scatter_add_(
            0, top1_indices, torch.ones(num_tokens, device=x.device, dtype=x.dtype)
        )
        f_i = expert_counts / num_tokens  # (num_experts,)

        # p_i: mean router probability per expert
        p_i = router_probs.mean(dim=0)  # (num_experts,)

        balance_loss = self.num_experts * (f_i * p_i).sum()

        # --- Energy loss ---
        #
        # L_energy = mean energy cost of the experts that were actually selected.
        # This gives a training-time gradient signal: "prefer routing patterns
        # that pick low-energy experts."
        #
        # Unlike the lambda penalty (which acts on logits at inference time),
        # beta * L_energy acts on the loss at training time, so it shapes W_g
        # during the entire training run.
        if e_gpu is not None:
            # Gather the energy cost for each token's selected expert(s)
            # e_gpu shape: (num_experts,) → index with top_k_indices
            selected_energy = e_gpu[top_k_indices]  # (num_tokens, top_k)
            energy_loss = selected_energy.mean()
        else:
            energy_loss = torch.tensor(0.0, device=x.device)

        # --- Router entropy ---
        #
        # H = -sum(p * log(p)) averaged over tokens.
        # Higher entropy = router is more uncertain = tokens are spread
        # across experts. Lower entropy = router is confident = tokens
        # concentrate on a few experts.
        #
        # This is a diagnostic metric, not a loss term.
        # We track it because:
        #   - Entropy decreasing over training → router is specializing (good)
        #   - Entropy near zero → expert collapse (bad, check balance loss)
        #   - Entropy near log(num_experts) → uniform routing (random-like)
        log_probs = torch.log(router_probs + 1e-10)  # avoid log(0)
        per_token_entropy = -(router_probs * log_probs).sum(dim=-1)
        router_entropy = per_token_entropy.mean()

        aux = {
            "balance_loss": balance_loss,
            "energy_loss": energy_loss,
            "router_entropy": router_entropy,
            "expert_load": f_i.detach(),  # for logging, not for gradients
        }

        return top_k_indices, top_k_weights, aux


class MoELayer(nn.Module):
    """Complete Mixture-of-Experts layer: gating + experts + dispatch.

    This replaces PositionwiseFFN in MoE transformer blocks.

    The dispatch logic handles the "scatter-gather" pattern:
      1. Router selects which expert processes each token (scatter)
      2. Each expert processes only its assigned tokens
      3. Outputs are combined using gate weights (gather)

    For k=1, each token goes to exactly one expert. The output is:
        output[t] = gate_weight[t, 0] * Expert_{index[t, 0]}(x[t])

    Since gate_weight is always 1.0 for k=1, this simplifies to:
        output[t] = Expert_{index[t, 0]}(x[t])

    For k=2, the output is a weighted combination:
        output[t] = w0 * Expert_{i0}(x[t]) + w1 * Expert_{i1}(x[t])

    Parameters
    ----------
    d_model, d_ff, num_experts, top_k, dropout, lambda_energy :
        See Expert and TopKGating for details.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 1,
        dropout: float = 0.1,
        lambda_energy: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # The gating network (router)
        self.gate = TopKGating(d_model, num_experts, top_k, lambda_energy)

        # The experts — nn.ModuleList so PyTorch tracks their parameters
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        e_gpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, d_model)
        e_gpu : Tensor, shape (num_experts,), optional
            Energy cost vector from the monitor.

        Returns
        -------
        output : Tensor, shape (batch, seq_len, d_model)
        aux : dict with balance_loss, energy_loss, router_entropy, expert_load
        """
        B, T, D = x.shape

        # Flatten batch and sequence for per-token routing
        x_flat = x.reshape(B * T, D)  # (num_tokens, d_model)

        # ── Route: decide which expert handles each token ──
        top_k_indices, top_k_weights, aux = self.gate(x_flat, e_gpu)
        # top_k_indices: (num_tokens, top_k) — expert index for each token
        # top_k_weights: (num_tokens, top_k) — gating weight for each selection

        # ── Dispatch: run each expert on its assigned tokens ──
        #
        # Naive implementation: loop over experts, mask tokens.
        # This is simple and correct. For a 4-expert testbed it's fine.
        # At scale (64+ experts), you'd use torch.scatter / all-to-all collectives.
        output = torch.zeros_like(x_flat)  # (num_tokens, d_model)

        for expert_idx in range(self.num_experts):
            # Find all (token, k) pairs where this expert was selected
            # mask shape: (num_tokens, top_k), True where expert_idx was chosen
            mask = (top_k_indices == expert_idx)

            if not mask.any():
                continue  # No tokens routed to this expert — skip

            # For each token, check if this expert appears in ANY of its top-k slots
            token_mask = mask.any(dim=-1)  # (num_tokens,) — True if token uses this expert

            # Get the tokens assigned to this expert
            expert_input = x_flat[token_mask]  # (num_assigned, d_model)

            # Run the expert
            expert_output = self.experts[expert_idx](expert_input)  # (num_assigned, d_model)

            # Accumulate weighted output
            # For each top-k slot where this expert appears, add:
            #   gate_weight * expert_output
            for k_idx in range(self.top_k):
                slot_mask = mask[:, k_idx]  # (num_tokens,) — True if this expert is in slot k_idx
                if not slot_mask.any():
                    continue
                # Weight for this slot
                weights = top_k_weights[slot_mask, k_idx].unsqueeze(-1)  # (n, 1)
                # Map slot_mask tokens to their position in expert_input
                # Since token_mask is a superset of slot_mask, we need the mapping
                token_indices = torch.where(token_mask)[0]
                slot_indices = torch.where(slot_mask)[0]
                # Find positions of slot_indices within token_indices
                # For k=1, slot_mask == token_mask, so this is an identity mapping
                pos_in_expert = torch.searchsorted(token_indices, slot_indices)
                output[slot_mask] += weights * expert_output[pos_in_expert]

        # Reshape back to (batch, seq_len, d_model)
        output = output.reshape(B, T, D)
        return output, aux

    def set_lambda_energy(self, lambda_energy: float) -> None:
        """Update the energy penalty coefficient at runtime.

        Useful for:
          - Sweeping lambda values without reloading the model
          - Curriculum strategies (start with lambda=0, ramp up over training)
        """
        self.gate.lambda_energy = lambda_energy
