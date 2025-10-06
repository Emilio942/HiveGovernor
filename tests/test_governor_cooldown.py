import torch

from swarm_moi.governor import Governor
from swarm_moi.router import HysteresisRouter


def test_governor_cooldown_counts_down_and_gates():
    n = 4
    router = HysteresisRouter(n_experts=n, k=1)
    gov = Governor(router=router, cooldown=2, gamma=0.5)

    pairs = [(1, 2, 5.0)]

    # First application: both indices inhibited
    inh1 = gov.apply(pairs)
    assert torch.isclose(inh1[1], torch.tensor(0.5))
    assert torch.isclose(inh1[2], torch.tensor(0.5))

    # Immediate second application should not further reduce due to cooldown
    inh2 = gov.apply(pairs)
    assert torch.allclose(inh2, inh1)

    # Tick without pairs to let cooldown reach zero
    _ = gov.apply([])  # cooldowns decrease from 1 to 0 (since we decrement first)

    # Now another apply should reduce again by gamma
    inh3 = gov.apply(pairs)
    assert torch.isclose(inh3[1], torch.tensor(0.25))
    assert torch.isclose(inh3[2], torch.tensor(0.25))

