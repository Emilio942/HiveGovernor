import torch

from swarm_moi.monitor import Monitor


def test_monitor_single_step_counts_with_ema_one():
    n = 3
    mon = Monitor(n_experts=n, ema=1.0)

    # Build batch mask (B=4)
    m = torch.tensor(
        [
            [1, 1, 0],  # co: (0,1)
            [1, 1, 0],  # co: (0,1)
            [0, 1, 1],  # co: (1,2)
            [0, 0, 1],  # co: (2)
        ],
        dtype=torch.float32,
    )

    C = mon.step(m)

    # Expected raw counts m^T m
    # diag: [2, 3, 2]
    # off-diag: C[0,1]=2, C[1,2]=1, others 0 (symmetric)
    expected = torch.tensor(
        [
            [2.0, 2.0, 0.0],
            [2.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    )
    assert torch.allclose(C, expected)


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

    # EMA formula starting from zeros:
    # after step1: C = 0.5 * C1
    # after step2: C = 0.5 * (0.5*C1) + 0.5 * C2 = 0.25*C1 + 0.5*C2
    expected = 0.25 * C1 + 0.5 * C2
    assert torch.allclose(C, expected)

