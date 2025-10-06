import torch

from swarm_moi.governor import Governor
from swarm_moi.monitor import Monitor
from swarm_moi.router import HysteresisRouter


def _identity_linear(n: int):
    """Helper to create an identity linear layer for predictable routing."""
    lin = torch.nn.Linear(n, n, bias=True)
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight.diagonal().fill_(1.0)
        lin.bias.zero_()
    return lin


def test_governor_breaks_ping_pong_pattern():
    """
    End-to-end test: Synthese-Setup with inputs that alternately trigger i↔j.
    Expectation: After ≤L steps, ping-pong disappears (inhibition takes effect).
    """
    # >>> BEGIN:AI_EDIT
    n_experts, k = 4, 2
    torch.manual_seed(42)  # For reproducible behavior
    
    # Set up components with more aggressive settings
    router = HysteresisRouter(n_experts=n_experts, k=k, tau=1.0, hysteresis=0.0)
    router.proj = _identity_linear(n_experts)
    
    monitor = Monitor(n_experts=n_experts, ema=0.5)
    governor = Governor(router=router, cooldown=1, gamma=0.4)
    
    # Create ping-pong inputs: alternately favor different pairs to create co-activations
    # Input 1 strongly favors experts 0 and 1 (will be selected together due to k=2)
    # Input 2 strongly favors experts 0 and 1 again (creating repeated co-activation)
    inputs_pingpong = [
        torch.tensor([[3.0, 2.5, -1.0, -1.0]], dtype=torch.float32),  # favors experts 0,1
        torch.tensor([[2.5, 3.0, -1.0, -1.0]], dtype=torch.float32),  # favors experts 1,0
    ]
    
    selections = []
    loop_pairs_history = []
    L = 12  # Maximum steps to break ping-pong
    
    for step in range(L):
        # Alternate between the two ping-pong inputs
        x = inputs_pingpong[step % 2]
        
        # Forward pass through router
        probs, mask = router(x)
        selected_experts = mask[0].nonzero(as_tuple=True)[0].tolist()
        selections.append(selected_experts)
        
        # Update monitor with selection mask
        C = monitor.step(mask)
        
        # Detect loop pairs - look for strong co-activation between experts 0 and 1
        loop_pairs = monitor.loop_pairs(thresh=0.1)  # Low threshold for easier detection
        loop_pairs_history.append(loop_pairs)
        
        # Apply governor to inhibit detected loops
        if loop_pairs:
            governor.apply(loop_pairs)
        
        # Check if we've broken the ping-pong pattern
        if step >= 4:  # Need several steps to build up co-activation and detect it
            # Check if the same pair {0,1} keeps being selected
            recent_selections = selections[-3:]
            all_same_pair = all(
                set(sel) == set(recent_selections[0]) 
                for sel in recent_selections
            )
            # If selections are no longer the same pair, ping-pong is broken
            if not all_same_pair:
                break
    else:
        # If we reached L steps without breaking, that's a failure
        assert False, (
            f"Ping-pong pattern not broken after {L} steps.\n"
            f"Selections: {selections}\n"
            f"Loop pairs detected: {loop_pairs_history}\n"
            f"Final inhibit: {router.inhibit.tolist()}\n"
            f"Final C matrix: {C.tolist()}"
        )
    
    # Verify that ping-pong was indeed broken within reasonable time
    assert step < L, f"Test should have broken ping-pong before step {L}"
    
    # Verify that some inhibition occurred during the process
    assert any(len(lp) > 0 for lp in loop_pairs_history), "Expected loop pairs to be detected at some point"
    assert torch.any(router.inhibit < 1.0), "Expected some experts to be inhibited"
    
    # Additional verification: co-activation matrix should show strong (0,1) pair
    assert C[0, 1] > 0, "Expected co-activation between experts 0 and 1"
    assert C[1, 0] > 0, "Expected symmetric co-activation"
    # >>> END:AI_EDIT
