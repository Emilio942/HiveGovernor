import torch

from swarm_moi.monitor import Monitor


def test_loop_pairs_relative_threshold_and_sorting():
    n = 4
    mon = Monitor(n_experts=n, ema=1.0)
    # Construct a symmetric C with strong pairs (0,1)=10 and (2,3)=8
    C = torch.tensor(
        [
            [5.0, 10.0, 1.0, 0.0],
            [10.0, 7.0, 0.5, 0.0],
            [1.0, 0.5, 6.0, 8.0],
            [0.0, 0.0, 8.0, 4.0],
        ]
    )
    mon._C = C

    # Threshold at 0.7 * max_off (max_off=10 -> t=7) => both pairs should pass
    pairs = mon.loop_pairs(thresh=0.7)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1), (2, 3)]
    assert [round(s, 3) for _, _, s in pairs] == [10.0, 8.0]


def test_loop_pairs_absolute_threshold_filters():
    n = 4
    mon = Monitor(n_experts=n, ema=1.0)
    C = torch.tensor(
        [
            [5.0, 10.0, 1.0, 0.0],
            [10.0, 7.0, 0.5, 0.0],
            [1.0, 0.5, 6.0, 8.0],
            [0.0, 0.0, 8.0, 4.0],
        ]
    )
    mon._C = C

    # Absolute threshold 9 -> only (0,1)
    pairs = mon.loop_pairs(thresh=9.0)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert pairs[0][2] == 10.0

