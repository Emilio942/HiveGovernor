import torch
import torch.nn.functional as F
from swarm_moi.router import HysteresisRouter
from swarm_moi.monitor import Monitor

def run_benchmark():
    print("HiveGovernor 'Proof of Value' (Integrated Additive Approach)")
    print("-" * 60)

    n_experts = 8
    n_dim = 128
    batch_size = 32
    steps = 100

    # Scenario A: Standard Router (without Integrated Additive Penalty)
    router_a = HysteresisRouter(n_experts, k=2, lambda_tax=0.0)
    monitor_a = Monitor(n_experts, ema=0.1)
    
    # Scenario B: Integrated Additive Router (with λ=0.04)
    router_b = HysteresisRouter(n_experts, k=2, lambda_tax=0.04)
    monitor_b = Monitor(n_experts, ema=0.1)

    # Force Bias in both routers to simulate stress
    biased_input = torch.randn(batch_size, n_dim)
    for r in [router_a, router_b]:
        r.eval()
        r(biased_input)
        r.train()
        with torch.no_grad():
            # Force experts 0 and 1 to be 'dominant' initially
            r.proj.weight.data[0:2, :] *= 10.0

    div_a, div_b = [], []

    for _ in range(steps):
        # Scenario A
        _, mask_a = router_a(biased_input)
        monitor_a.step(mask_a)
        div_a.append(router_a.adaptive_capacity(mask_a))

        # Scenario B
        _, mask_b = router_b(biased_input)
        monitor_b.step(mask_b)
        div_b.append(router_b.adaptive_capacity(mask_b))

    avg_div_a = sum(div_a) / steps
    avg_div_b = sum(div_b) / steps
    
    b_score_a = monitor_a.bottleneck_score()
    b_score_b = monitor_b.bottleneck_score()

    print(f"Standard MoE Diversity: {avg_div_a:.4f}")
    print(f"Integrated Additive Diversity: {avg_div_b:.4f}")
    print(f"Improvement: {((avg_div_b/avg_div_a)-1)*100:.2f}%")
    print("-" * 60)
    print(f"Standard MoE Bottleneck Score: {b_score_a:.4f}")
    print(f"Integrated Additive Bottleneck Score: {b_score_b:.4f}")
    print(f"Robustness Gain: {((b_score_b/(b_score_a+1e-6))-1)*100:.2f}%")

if __name__ == "__main__":
    run_benchmark()
