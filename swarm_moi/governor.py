from __future__ import annotations

from typing import Any, Sequence, Tuple

import torch


class Governor:
    """
    Governor applies multiplicative inhibition to router experts based on detected loop pairs,
    while enforcing a per-expert cooldown (refractory) period.
    """

    def __init__(self, router: Any, cooldown: int = 2, gamma: float = 0.5) -> None:
        # >>> BEGIN:AI_EDIT
        if cooldown < 0:
            raise ValueError("cooldown must be >= 0")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")

        if not hasattr(router, "n_experts"):
            raise ValueError("router must provide n_experts attribute")

        self.router = router
        self.cooldown = int(cooldown)
        self.gamma = float(gamma)
        self.n_experts: int = int(router.n_experts)

        # Ensure inhibit vector exists on the router (multiplicative factors)
        if not hasattr(self.router, "inhibit"):
            self.router.inhibit = torch.ones(self.n_experts, dtype=torch.float32)

        # Per-expert cooldown counters (steps remaining until re-eligible)
        self._cd = torch.zeros(self.n_experts, dtype=torch.int64)
        # >>> END:AI_EDIT

    def apply(
        self, loop_pairs: Sequence[Tuple[int, int, float]], top_k_pairs: int = 1
    ) -> torch.Tensor:
        """
        Apply inhibition to top loop pair(s) if their experts are not in cooldown.

        - Decrements cooldown counters by 1 (floored at 0) each call.
        - For up to `top_k_pairs` pairs, multiply the involved experts' inhibit factors by `gamma`
          if their cooldown is 0, and set their cooldown to `cooldown`.

        Returns the updated inhibition vector.
        """
        # >>> BEGIN:AI_EDIT
        # Decay cooldown counters
        if len(self._cd) > 0:
            self._cd = (self._cd - 1).clamp_min(0)

        inh = self.router.inhibit
        if not isinstance(inh, torch.Tensor) or inh.numel() != self.n_experts:
            inh = torch.ones(self.n_experts, dtype=torch.float32)
            self.router.inhibit = inh

        applied = 0
        for i, j, _score in loop_pairs:
            if applied >= top_k_pairs:
                break
            any_applied = False
            # Inhibit per expert if eligible
            for idx in (i, j):
                if 0 <= idx < self.n_experts and int(self._cd[idx].item()) == 0:
                    inh[idx] = (inh[idx] * self.gamma).clamp_min(1e-6)
                    self._cd[idx] = self.cooldown
                    any_applied = True
            if any_applied:
                applied += 1

        return inh.clone()
        # >>> END:AI_EDIT

    def cooldowns(self) -> torch.Tensor:
        return self._cd.clone()

