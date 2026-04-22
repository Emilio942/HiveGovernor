from __future__ import annotations

import torch
from swarm_moi.monitor import Monitor


def test_monitor_init():
    n = 8
    mon = Monitor(n_experts=n)
    C = mon.co_matrix()
    # Q16: Verify Bayesian Prior (1/N)
    assert torch.allclose(C, torch.ones((n, n)) / n)


def test_monitor_two_steps_ema_blend():
    n = 3
    ema = 0.5
    mon = Monitor(n_experts=n, ema=ema)

    m1 = torch.tensor(
        [
            [1, 0, 0],  # only 0
            [1, 0, 0],  # only 0
        ],
        dtype=torch.float32,
    )
    C1 = m1.T @ m1  # [[2,0,0],[0,0,0],[0,0,0]]
    mon.step(m1)

    m2 = torch.tensor(
        [
            [0, 1, 0],  # only 1
            [0, 1, 0],  # only 1
            [0, 1, 0],  # only 1
        ],
        dtype=torch.float32,
    )
    C2 = m2.T @ m2  # [[0,0,0],[0,3,0],[0,0,0]]
    C = mon.step(m2)

    # Q16: Updated EMA formula starting from Bayesian Prior (1/N)
    prior = torch.ones((n, n)) / n
    # after step1: C = 0.5 * prior + 0.5 * C1
    # after step2: C = 0.5 * (0.5 * prior + 0.5 * C1) + 0.5 * C2
    expected = 0.25 * prior + 0.25 * C1 + 0.5 * C2
    assert torch.allclose(C, expected)
