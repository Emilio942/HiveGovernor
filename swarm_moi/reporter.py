from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


def build_report(
    C: torch.Tensor,
    loop_pairs: List[Tuple[int, int, float]],
    labels: torch.Tensor,
    max_groups: int = 5,
    max_loop_pairs: int = 3
) -> Dict[str, Any]:
    """
    Build a structured report from co-activation matrix, loop pairs, and cluster labels.
    
    Args:
        C: Co-activation matrix of shape (n_experts, n_experts)
        loop_pairs: List of (expert1, expert2, strength) tuples from loop detection
        labels: Cluster labels of shape (n_experts,)
        max_groups: Maximum number of top groups to include
        max_loop_pairs: Maximum number of top loop pairs to include
        
    Returns:
        Dict with keys: 'top_groups', 'top_loop_pairs', 'inhibition_history', 'summary'
    """
    # >>> BEGIN:AI_EDIT
    n_experts = C.size(0) if C.numel() > 0 else 0
    
    # Build top groups analysis
    top_groups = _analyze_top_groups(labels, C, max_groups)
    
    # Build top loop pairs analysis
    top_loop_pairs = _analyze_loop_pairs(loop_pairs, max_loop_pairs)
    
    # Build inhibition history stub (placeholder for future extension)
    inhibition_history = _build_inhibition_history_stub()
    
    # Build summary statistics
    summary = _build_summary(n_experts, top_groups, top_loop_pairs)
    
    return {
        'top_groups': top_groups,
        'top_loop_pairs': top_loop_pairs, 
        'inhibition_history': inhibition_history,
        'summary': summary
    }


def _analyze_top_groups(
    labels: torch.Tensor, C: torch.Tensor, max_groups: int
) -> List[Dict[str, Any]]:
    """Analyze cluster groups and return top groups by size."""
    if labels.numel() == 0:
        return []
    
    unique_labels = torch.unique(labels)
    groups = []
    
    for label in unique_labels:
        mask = (labels == label)
        expert_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
        group_size = len(expert_indices)
        
        # Calculate average intra-group co-activation
        if group_size > 1 and C.numel() > 0:
            group_C = C[mask][:, mask]
            # Exclude diagonal for average
            off_diag = group_C - torch.diag(torch.diag(group_C))
            avg_coactivation = (
                off_diag.sum() / (group_size * (group_size - 1))
                if group_size > 1
                else 0.0
            )
        else:
            avg_coactivation = 0.0
        
        # Choose representative expert (highest total co-activation in group)
        if group_size > 0 and C.numel() > 0:
            group_totals = C[mask].sum(dim=1)
            best_idx = torch.argmax(group_totals)
            representative = expert_indices[best_idx]
        else:
            representative = expert_indices[0] if expert_indices else 0
        
        groups.append({
            'label': int(label.item()),
            'size': group_size,
            'experts': expert_indices,
            'representative': representative,
            'avg_coactivation': float(avg_coactivation)
        })
    
    # Sort by size (descending) and limit
    groups.sort(key=lambda x: x['size'], reverse=True)
    return groups[:max_groups]


def _analyze_loop_pairs(
    loop_pairs: List[Tuple[int, int, float]], max_pairs: int
) -> List[Dict[str, Any]]:
    """Analyze loop pairs and return top pairs by strength."""
    if not loop_pairs:
        return []
    
    analyzed_pairs = []
    for expert1, expert2, strength in loop_pairs[:max_pairs]:
        analyzed_pairs.append({
            'expert1': expert1,
            'expert2': expert2,
            'strength': float(strength),
            'pair_id': f"{min(expert1, expert2)}-{max(expert1, expert2)}"
        })
    
    return analyzed_pairs


def _build_inhibition_history_stub() -> Dict[str, Any]:
    """Build inhibition history stub for future extension."""
    return {
        'total_inhibitions': 0,
        'recent_inhibitions': [],
        'most_inhibited_experts': [],
        'note': 'Inhibition history tracking not yet implemented'
    }


def _build_summary(
    n_experts: int, top_groups: List[Dict[str, Any]], top_loop_pairs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build summary statistics."""
    total_groups = len(top_groups)
    largest_group_size = top_groups[0]['size'] if top_groups else 0
    total_loop_pairs = len(top_loop_pairs)
    strongest_loop_strength = top_loop_pairs[0]['strength'] if top_loop_pairs else 0.0
    
    return {
        'n_experts': n_experts,
        'total_groups': total_groups,
        'largest_group_size': largest_group_size,
        'total_loop_pairs': total_loop_pairs,
        'strongest_loop_strength': strongest_loop_strength,
        'fragmentation_ratio': total_groups / max(n_experts, 1)  # 1.0 = all isolated
    }
    # >>> END:AI_EDIT
