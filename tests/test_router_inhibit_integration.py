import torch

from swarm_moi.router import HysteresisRouter


def _identity_linear(n: int):
    lin = torch.nn.Linear(n, n, bias=True)
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight.diagonal().fill_(1.0)
        lin.bias.zero_()
    return lin


def test_inhibition_reduces_probability_and_changes_selection():
    n, k = 2, 1
    r = HysteresisRouter(n_experts=n, k=k, tau=1.0, hysteresis=0.0)
    r.proj = _identity_linear(n)

    x = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # equal logits -> equal probs
    p1, m1 = r(x)
    # By tie-breaking, top-k chooses index 0
    assert m1[0, 0].item() is True

    # Inhibit expert 0 strongly
    r.inhibit = torch.tensor([0.01, 1.0], dtype=torch.float32)
    p2, m2 = r(x)

    # Probability of expert 0 should be much lower than expert 1
    assert p2[0, 0] < p2[0, 1]
    # Selection should flip to expert 1
    assert m2[0, 1].item() is True

