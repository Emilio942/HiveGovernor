from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HysteresisRouter(nn.Module):
    """
    Integrated Additive Router (Final Stabilized Version).
    Features:
    - Exact Analytical Gradient (Logits-to-Prob Jacobian).
    - Logit Centering (Breaking row-wise additive invariance).
    - Momental Damping (Preventing Hopf Bifurcation / Router Jitter).
    - Convergent Sinkhorn-Knopp.
    """

    def __init__(
        self, 
        n_experts: int, 
        k: int, 
        tau: float = 1.0, 
        hysteresis: float = 0.0,
        lambda_tax: float = 0.04,
        alpha_damping: float = 0.7 # Q18/Stability: Damping factor for limit-cycle prevention
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.tau = tau
        self.hysteresis = hysteresis
        self.lambda_tax = lambda_tax
        self.alpha_damping = alpha_damping

        self.proj: nn.Linear = nn.LazyLinear(n_experts, bias=True)
        
        # Buffer for Damping/Momentum
        self.register_buffer("_prev_probs", torch.empty(0), persistent=False)
        self._prev_probs: torch.Tensor
        
        # Buffer for Hysteresis
        self.register_buffer("_prev_mask", torch.empty(0), persistent=False)
        self._prev_mask: torch.Tensor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(x)
        
        # 1. Break row-wise invariance (Ghost Governor Fix)
        logits = logits - logits.mean(dim=-1, keepdim=True)
        
        # 2. Initial Softmax
        M = F.softmax(logits / self.tau, dim=-1)

        # 3. Additive Tax (Integrated Optimization)
        if self.training and self.lambda_tax > 0:
            C = M.transpose(0, 1) @ M
            C.fill_diagonal_(0)
            
            # Exact Jacobian-vector product
            grad_m = 4.0 * (M @ C)
            exact_grad = (M * grad_m) - M * torch.sum(M * grad_m, dim=-1, keepdim=True)
            
            logits = logits - self.lambda_tax * exact_grad
            M = F.softmax(logits / self.tau, dim=-1)

        # 4. Convergent Sinkhorn-Knopp (Global Balance)
        if self.training:
            for _ in range(10): # Increased iterations for true DS property
                M = M / M.sum(dim=0, keepdim=True).clamp_min(1e-12)
                M = M * (self.n_experts / M.size(0))
                M = M / M.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # 5. Momental Damping (Limit Cycle Prevention)
        if self.training and self._prev_probs.numel() > 0:
            # Current mapping f(Z) blended with previous state
            M = (1.0 - self.alpha_damping) * self._prev_probs + self.alpha_damping * M
        
        self._prev_probs = M.detach()

        # 6. Hysteresis (Temporal Stability)
        if self._prev_mask.numel() > 0 and self.hysteresis > 0.0:
            prev = self._prev_mask.to(dtype=M.dtype)
            denom = prev.sum(dim=-1, keepdim=True).clamp_min(1.0)
            M = (1.0 - self.hysteresis) * M + self.hysteresis * (prev / denom)
            M = M / M.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # 7. Top-k selection
        topk = torch.topk(M, k=self.k, dim=-1).indices
        mask = torch.zeros_like(M, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk, value=True)
        self._prev_mask = mask.detach()

        return M, mask

    def adaptive_capacity(self, mask: torch.Tensor) -> float:
        freq = mask.float().mean(dim=0)
        p = freq / (freq.sum() + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
        max_entropy = torch.log(torch.tensor(float(self.n_experts))).item()
        return entropy / (max_entropy + 1e-12)
