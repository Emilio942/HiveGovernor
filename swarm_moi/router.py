from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HysteresisRouter(nn.Module):
    """
    Minimal router with temperature softmax and hysteresis over last selection mask.

    Args:
        n_experts: number of experts
        k: number of experts to select (top-k)
        tau: softmax temperature (>0)
        hysteresis: blend factor in [0,1]; 0 disables hysteresis, 1 keeps last selection
    Returns (forward):
        probs: (B, n_experts) selection probabilities after hysteresis blend
        mask:  (B, n_experts) boolean mask with exactly k True per row
    """

    def __init__(self, n_experts: int, k: int, tau: float = 1.0, hysteresis: float = 0.0) -> None:
        super().__init__()
        if not (1 <= k <= n_experts):
            raise ValueError("k must satisfy 1 <= k <= n_experts")
        if tau <= 0:
            raise ValueError("tau must be > 0")
        if not (0.0 <= hysteresis <= 1.0):
            raise ValueError("hysteresis must be in [0, 1]")

        self.n_experts = n_experts
        self.k = k
        self.tau = float(tau)
        self.hysteresis = float(hysteresis)

        # Placeholder projector; lazily infers input dimension on first forward
        self.proj: nn.Linear = nn.LazyLinear(n_experts, bias=True)

        # buffer to hold previous selection mask
        self.register_buffer("_prev_mask", torch.empty(0), persistent=False)
        self._prev_mask: torch.Tensor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # >>> BEGIN:AI_EDIT
        logits = self.proj(x)  # (B, n_experts)
        probs = F.softmax(logits / self.tau, dim=-1)

        if self._prev_mask.numel() > 0 and self.hysteresis > 0.0:
            prev = self._prev_mask.to(dtype=probs.dtype)
            denom = prev.sum(dim=-1, keepdim=True).clamp_min(1.0)
            prev_probs = prev / denom
            probs = (1.0 - self.hysteresis) * probs + self.hysteresis * prev_probs
            # Re-normalize to remain a proper distribution
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # Apply multiplicative inhibition hook if present, then renormalize
        inh = getattr(self, "inhibit", None)
        if isinstance(inh, torch.Tensor) and inh.numel() == self.n_experts:
            inh = inh.to(device=probs.device, dtype=probs.dtype)
        else:
            inh = torch.ones(self.n_experts, device=probs.device, dtype=probs.dtype)
        probs = probs * inh.unsqueeze(0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # Top-k selection
        topk = torch.topk(probs, k=self.k, dim=-1).indices
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk, value=True)

        # cache for next step
        self._prev_mask = mask.detach()

        return probs, mask
        # >>> END:AI_EDIT
