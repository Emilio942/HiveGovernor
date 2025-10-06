import torch

from swarm_moi.router import HysteresisRouter


def test_router_init_and_shapes():
    torch.manual_seed(0)
    n_experts, k = 5, 2
    router = HysteresisRouter(n_experts=n_experts, k=k, tau=0.7, hysteresis=0.3)

    B, D = 4, 8
    x = torch.randn(B, D)
    probs, mask = router(x)

    assert probs.shape == (B, n_experts)
    assert mask.shape == (B, n_experts)
    assert mask.dtype == torch.bool
    # exactly k selections per sample
    assert torch.all(mask.sum(dim=-1) == k)
    # probs form valid distributions
    s = probs.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6)
    assert torch.all(probs >= 0)

