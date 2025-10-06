import torch

from swarm_moi.governor import Governor
from swarm_moi.router import HysteresisRouter


def test_governor_applies_gamma_inhibition():
    n = 5
    router = HysteresisRouter(n_experts=n, k=1)
    gov = Governor(router=router, cooldown=1, gamma=0.5)

    # Initially inhibition vector is all ones
    assert torch.allclose(router.inhibit, torch.ones(n))

    pairs = [(1, 3, 10.0)]
    inh = gov.apply(pairs)

    expected = torch.ones(n)
    expected[1] = 0.5
    expected[3] = 0.5

    assert torch.allclose(inh, expected)
    assert torch.allclose(router.inhibit, expected)

