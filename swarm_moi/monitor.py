from __future__ import annotations

import torch


class Monitor:
    """
    Tracks co-activation matrix C using an exponential moving average.
    Verified Version:
    - Q16: Bayesian Prior Initialization (1/N uniform) to avoid cold-start bias.
    - Q4: Spectral Floor Correction (Lollipop Artifact Neutralization).
    """

    def __init__(self, n_experts: int, ema: float = 0.1) -> None:
        if not (0.0 < ema <= 1.0):
            raise ValueError("ema must be in (0, 1]")
        self.n_experts = int(n_experts)
        self.ema = float(ema)
        
        # Q16: Bayesian Prior Initialization
        # Instead of zero, start with a uniform weak prior (1/N)
        self._C = torch.ones((n_experts, n_experts), dtype=torch.float32) / n_experts

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
        Q4: Corrected Bottleneck Score.
        Uses Fiedler value with (1 - sigma) correction to remove 
        Sinkhorn-induced spectral inflation (Lollipop Artifact).
        """
        C = self._C
        if C.numel() == 0 or self.n_experts < 2:
            return 1.0

        A = torch.abs(C)
        A.fill_diagonal_(0)
        
        # Q4: Calculate row-variance sigma as correction factor
        row_sums = A.sum(dim=1)
        if (row_sums == 0).all():
            return 0.0
            
        # Normalized variance of row sums (deviation from uniform)
        mean_row = row_sums.mean()
        sigma = torch.sqrt(torch.mean((row_sums - mean_row)**2)) / (mean_row + 1e-12)
        
        # Standard Laplacian calculation
        d_inv_sqrt = torch.pow(row_sums.clamp(min=1e-6), -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        L = torch.eye(self.n_experts, device=C.device) - D_inv_sqrt @ A @ D_inv_sqrt
        
        try:
            eigenvals = torch.linalg.eigvalsh(L)
            lambda_2 = eigenvals[1].item()
            # Q4 Correction: (lambda_2 / 2) * (1 - sigma)
            # This 'deflates' the artificially inflated spectral gap.
            return (lambda_2 / 2.0) * (1.0 - sigma.item())
        except (RuntimeError, IndexError):
            return 0.0
