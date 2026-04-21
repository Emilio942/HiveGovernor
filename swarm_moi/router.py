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

    def __init__(
        self, 
        n_experts: int, 
        k: int, 
        tau: float = 1.0, 
        hysteresis: float = 0.0,
        langevin_tau: float = 0.0 # Q6: Langevin noise temperature
    ) -> None:
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
        self.langevin_tau = float(langevin_tau)

        # Placeholder projector; lazily infers input dimension on first forward
        self.proj: nn.Linear = nn.LazyLinear(n_experts, bias=True)
        
        # buffer to hold previous selection mask
        self.register_buffer("_prev_mask", torch.empty(0), persistent=False)
        self._prev_mask: torch.Tensor
        self._hook_registered = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # >>> BEGIN:AI_EDIT
        # Q4: Natural Gradient Hook (Delayed for LazyLinear)
        if not self._hook_registered and not isinstance(self.proj.weight, torch.nn.parameter.UninitializedParameter):
            def natural_grad_hook(grad):
                return grad - grad.mean(dim=-1, keepdim=True)
            self.proj.weight.register_hook(natural_grad_hook)
            self._hook_registered = True

        logits = self.proj(x)  # (B, n_experts)
        
        # Q6: Langevin Stochastic Routing
        # Add thermal noise to logits to guarantee exploration and prevent "starvation"
        if self.training and self.langevin_tau > 0:
            noise = torch.randn_like(logits) * self.langevin_tau
            logits = logits + noise

        probs = F.softmax(logits / self.tau, dim=-1)


        # Q3: Sinkhorn-Knopp Balancing (Ensuring balanced utilization)
        # We iteratively normalize rows and columns to make the matrix nearly doubly stochastic.
        if self.training:
            for _ in range(3): # 3 iterations are usually enough for stability
                # Column normalization (balance across experts)
                probs = probs / probs.sum(dim=0, keepdim=True).clamp_min(1e-12)
                probs = probs * (self.n_experts / probs.size(0)) # Scale to match batch size
                # Row normalization (valid distribution per sample)
                probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

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

    def adaptive_capacity(self, mask: torch.Tensor) -> float:
        """
        Estimate adaptive capacity via Shannon entropy of selection counts (Q11).
        High entropy indicates the system is near the 'edge of chaos'.
        """
        if mask.numel() == 0:
            return 0.0
        
        # Expert usage frequency in batch
        freq = mask.float().mean(dim=0)
        # Normalize to probability distribution
        p = freq / (freq.sum() + 1e-12)
        # Entropy H = -sum(p * log(p))
        entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
        
        # Normalized entropy (0 to 1)
        max_entropy = torch.log(torch.tensor(float(self.n_experts))).item()
        return entropy / (max_entropy + 1e-12)

