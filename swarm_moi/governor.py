from __future__ import annotations

from typing import Any, Sequence, Tuple
import torch

class Governor:
    """
    Governor applies inhibition to router experts based on detected loop pairs.
    """

    def __init__(self, router: Any, gamma: float = 0.5) -> None:
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        if not hasattr(router, "n_experts"):
            raise ValueError("router must provide n_experts attribute")

        self.router = router
        self.gamma = float(gamma)
        self.n_experts: int = int(router.n_experts)

        if not hasattr(self.router, "inhibit"):
            self.router.inhibit = torch.ones(self.n_experts, dtype=torch.float32)

    def apply(self, loop_pairs: Sequence[Tuple[int, int, float]]) -> torch.Tensor:
        """
        Apply simple heuristic inhibition to experts in loop pairs.
        """
        inh = self.router.inhibit
        if not isinstance(inh, torch.Tensor) or inh.numel() != self.n_experts:
            inh = torch.ones(self.n_experts, dtype=torch.float32)
            self.router.inhibit = inh

        for i, j, _score in loop_pairs:
            for idx in (i, j):
                if 0 <= idx < self.n_experts:
                    inh[idx] = (inh[idx] * self.gamma).clamp_min(1e-6)

        return inh.clone()
