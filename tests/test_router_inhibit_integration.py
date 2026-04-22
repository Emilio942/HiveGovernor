from __future__ import annotations

import torch
import torch.nn as nn
from swarm_moi.router import HysteresisRouter


def _identity_linear(n: int) -> nn.Linear:
    proj = nn.Linear(n, n)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(n))
        proj.bias.zero_()
    return proj


def test_additive_tax_minimizes_coactivation_energy():
    """
    Verifies that the Integrated Additive Tax (lambda_tax) 
    reduces the co-activation energy P(M) = ||M^T M - diag||^2.
    """
    n_experts, k = 2, 1
    batch_size = 4
    
    # We use asymmetrical logits to give the tax something to optimize
    # Sample 0,1 prefer expert 0. Sample 2,3 prefer expert 0 too.
    # But tax should push them apart.
    x = torch.tensor([[1.0, 0.0], [1.1, 0.0], [0.9, 0.0], [1.0, 0.1]], dtype=torch.float32)
    
    # Enable lambda_tax
    r = HysteresisRouter(n_experts=n_experts, k=k, tau=1.0, lambda_tax=50.0) # High tax for clear effect
    r.proj = _identity_linear(n_experts)
    r.train() 

    # Reference without tax
    r_no_tax = HysteresisRouter(n_experts=n_experts, k=k, tau=1.0, lambda_tax=0.0)
    r_no_tax.proj = _identity_linear(n_experts)
    r_no_tax.train()
    
    p_no_tax, _ = r_no_tax(x)
    p_tax, _ = r(x)

    def co_activation_energy(M):
        C = M.transpose(0, 1) @ M
        C.fill_diagonal_(0)
        return torch.norm(C, p='fro').item()

    energy_no_tax = co_activation_energy(p_no_tax)
    energy_tax = co_activation_energy(p_tax)

    # The tax is explicitly designed to minimize this energy.
    # Energy with tax MUST be lower than without tax.
    assert energy_tax < energy_no_tax
