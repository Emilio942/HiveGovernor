from __future__ import annotations

import torch


class Monitor:
    """
    Tracks co-activation matrix C using an exponential moving average.
    """

    def __init__(self, n_experts: int, ema: float = 1.0) -> None:
        if not (0.0 < ema <= 1.0):
            raise ValueError("ema must be in (0, 1]")
        self.n_experts = int(n_experts)
        self.ema = float(ema)
        self._C = torch.zeros((n_experts, n_experts), dtype=torch.float32)

    def step(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() != 2 or mask.size(-1) != self.n_experts:
            raise ValueError("mask must have shape (B, n_experts)")

        if self._C.device != mask.device:
            self._C = self._C.to(mask.device)

        m = mask.to(dtype=torch.float32)
        batch_C = m.transpose(0, 1) @ m

        # EMA update
        self._C.mul_(1.0 - self.ema).add_(batch_C, alpha=self.ema)

        return self._C

    def co_matrix(self) -> torch.Tensor:
        return self._C.clone()

    def bottleneck_score(self) -> float:
        """
        Q10: Approximation of the Cheeger constant.
        Low score = bottleneck (failure). High score = well-connected (success).
        """
        C = self._C
        if C.numel() == 0 or self.n_experts < 2:
            return 1.0

        A = torch.abs(C)
        A.fill_diagonal_(0)
        d = A.sum(dim=1)
        
        if (d == 0).all():
            return 0.0
            
        d_inv_sqrt = torch.pow(d.clamp(min=1e-6), -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        L = torch.eye(self.n_experts, device=C.device) - D_inv_sqrt @ A @ D_inv_sqrt
        
        try:
            eigenvals = torch.linalg.eigvalsh(L)
            lambda_2 = eigenvals[1].item()
            return lambda_2 / 2.0
        except (RuntimeError, IndexError):
            return 0.0
