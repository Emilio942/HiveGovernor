import torch

from swarm_moi.router import HysteresisRouter


def _identity_linear(n: int):
    lin = torch.nn.Linear(n, n, bias=True)
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight.diagonal().fill_(1.0)
        lin.bias.zero_()
    return lin


def test_hysteresis_stabilizes_selection_under_small_change():
    torch.manual_seed(0)
    n, k = 4, 1

    # No hysteresis
    r0 = HysteresisRouter(n_experts=n, k=k, tau=1.0, hysteresis=0.0)
    r0.proj = _identity_linear(n)

    # With hysteresis
    r1 = HysteresisRouter(n_experts=n, k=k, tau=1.0, hysteresis=0.8)
    r1.proj = _identity_linear(n)

    # Step 1: pick expert 0 as top
    x1 = torch.tensor([[1.00, 0.99, 0.0, 0.0]], dtype=torch.float32)
    _, m0_t1 = r0(x1)
    _, m1_t1 = r1(x1)

    # Step 2: slight change favors expert 1 by a hair
    x2 = torch.tensor([[0.995, 1.000, 0.0, 0.0]], dtype=torch.float32)
    _, m0_t2 = r0(x2)
    _, m1_t2 = r1(x2)

    # Without hysteresis, selection should flip to expert 1
    assert not torch.equal(m0_t1, m0_t2)

    # With strong hysteresis, selection should remain stable (stick to expert 0)
    assert torch.equal(m1_t1, m1_t2)

