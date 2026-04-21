# HiveGovernor: Integrated Additive MoE Routing

HiveGovernor is a mathematically rigorous management system for Mixture of Experts (MoE) architectures. It prevents **Expert Collapse** and **Redundant Co-activation** by integrating structural penalties directly into the routing manifold.

## The Core Problem
Standard MoE systems often suffer from experts converging on the same features, reducing system capacity. Simple "handbrake" methods (multiplicative inhibition) fail because normalization layers (like Sinkhorn-Knopp) are scale-invariant and effectively neutralize the control signal.

## The Solution: Integrated Additive Optimization
Unlike heuristic governors, HiveGovernor uses an **Integrated Additive Approach**:
1. **Logit Centering:** Breaks row-wise additive invariance (eliminating the 'Ghost Governor' effect).
2. **Exact Gradient Penalty:** Uses the analytical derivative of the co-activation energy, including the Softmax-Jacobian-Vector product.
3. **Sinkhorn-Knopp Manifold:** Enforces doubly-stochastic constraints for global load balancing.
4. **Momental Damping:** Prevents limit-cycle oscillations (Router Jitter) via inertial stabilization.

## Mathematical Foundation
The system solves a coupled optimization problem in every forward pass:
$$\min_{M \in \Delta_{DS}} \text{KL}(M \| \text{softmax}(Z)) + \lambda_{tax} \| M^T M - \text{diag}(M^T M) \|_F^2$$

## Getting Started
### Dependencies
- Python 3.12+
- PyTorch

### Running the Proof of Value
To verify the system's ability to maintain diversity under high bias stress, run:
```bash
export PYTHONPATH=$PYTHONPATH:.
python tools/proof_of_value.py
```

## Documentation
- `plans/MATHEMATICAL_DECONSTRUCTION.md`: Formal proof of the neutralization theorem and the lollipop artifact audit.
