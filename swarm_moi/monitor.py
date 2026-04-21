from __future__ import annotations

import torch


class Monitor:
    """
    Tracks co-activation matrix C using an exponential moving average.

    C <- (1 - ema) * C + ema * (m^T m)

    Where m is a batch mask of shape (B, n_experts), typically boolean or {0,1}.
    """

    def __init__(self, n_experts: int, ema: float = 1.0) -> None:
        if not (0.0 < ema <= 1.0):
            raise ValueError("ema must be in (0, 1]")
        self.n_experts = int(n_experts)
        self.ema = float(ema)
        self._device = torch.device("cpu")
        self._C = torch.zeros((n_experts, n_experts), dtype=torch.float32)

    def step(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Update co-activation matrix given a batch selection mask.
        """
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

    def loop_pairs(self, thresh: float) -> list[tuple[int, int, float]]:
        """
        Detect strong symmetric co-activation pairs (i < j).
        """
        C = self._C
        if C.numel() == 0:
            return []

        A = C.clone()
        A.fill_diagonal_(0)

        max_off = A.max().item() if A.numel() > 0 else 0.0
        t = thresh * max_off if 0.0 <= thresh <= 1.0 else float(thresh)

        A_triu = torch.triu(A, diagonal=1)
        idx = (A_triu >= t).nonzero(as_tuple=False)
        pairs = [(int(i), int(j), float(A[i, j].item())) for i, j in idx]
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs
