from __future__ import annotations

import torch


def cluster_from_C(
    C: torch.Tensor, method: str = "spectral", threshold: float = 0.1
) -> torch.Tensor:
    """
    Cluster experts based on co-activation matrix C.
    
    Args:
        C: Co-activation matrix of shape (n_experts, n_experts)
        method: Clustering method - "spectral" or "threshold"
        threshold: For threshold method, minimum connection strength to be in same cluster
        
    Returns:
        labels: Tensor of shape (n_experts,) with cluster assignments (0, 1, 2, ...)
    """
    # >>> BEGIN:AI_EDIT
    if C.dim() != 2 or C.size(0) != C.size(1):
        raise ValueError("C must be a square matrix")
    
    if method == "spectral":
        return _spectral_clustering(C)
    elif method == "threshold":
        return _threshold_clustering(C, threshold)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def _spectral_clustering(C: torch.Tensor) -> torch.Tensor:
    """
    Spectral clustering using the Eigengap heuristic to determine k automatically.
    Uses the symmetrically normalized Laplacian.
    """
    n = C.size(0)
    if n <= 1:
        return torch.zeros(n, dtype=torch.long)
    
    # Adjacency matrix (absolute co-activations, no self-loops)
    A = torch.abs(C)
    A.fill_diagonal_(0)
    
    # Degree matrix
    d = A.sum(dim=1)
    if (d == 0).all():
        return torch.arange(n, dtype=torch.long)
    
    # Symmetrically Normalized Laplacian: L = I - D^-1/2 A D^-1/2
    d_inv_sqrt = torch.pow(d.clamp(min=1e-6), -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    L_norm = torch.eye(n, device=C.device, dtype=C.dtype) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Eigen-decomposition
    try:
        eigenvals, eigenvecs = torch.linalg.eigh(L_norm)
    except RuntimeError:
        return torch.zeros(n, dtype=torch.long)
    
    # Eigengap Heuristic: Find max gap between consecutive eigenvalues
    # Skip the first (always 0) and look for the gap that separates clusters
    if n > 2:
        gaps = eigenvals[1:] - eigenvals[:-1]
        # Look for the maximum gap in the first half of spectrum (most stable k)
        max_gap_idx = torch.argmax(gaps[1:n//2 + 1])
        k = max_gap_idx + 2 
    else:
        k = 2

    # Use the first k eigenvectors for clustering
    # We use a simple k-means proxy: sign patterns of the first k-1 non-trivial vectors
    selected_vecs = eigenvecs[:, 1:int(k)]
    
    if int(k) == 2:
        # Binary split using the Fiedler vector
        fiedler = eigenvecs[:, 1]
        # Use sign if it clearly separates, otherwise median
        if (fiedler > 0).any() and (fiedler < 0).any():
            labels = (fiedler >= 0).long()
        else:
            labels = (fiedler >= torch.median(fiedler)).long()
    else:
        # For k > 2, create a bit-mask from signs to separate groups
        # This is a robust way to turn eigenvector signs into cluster IDs
        bits = (selected_vecs > 0).long()
        powers = torch.pow(2, torch.arange(bits.size(1), device=C.device))
        labels = (bits * powers).sum(dim=1)
        # Re-map to 0..k-1 range
        unique_labels = torch.unique(labels)
        for i, val in enumerate(unique_labels):
            labels[labels == val] = i
        
    return labels




def _threshold_clustering(C: torch.Tensor, threshold: float) -> torch.Tensor:
    """Threshold-based clustering: experts with co-activation >= threshold are in same cluster."""
    n = C.size(0)
    
    # Create binary adjacency matrix
    A = (torch.abs(C) >= threshold).float()
    
    # Find connected components using a simple flood-fill approach
    labels = torch.full((n,), -1, dtype=torch.long)
    current_label = 0
    
    for i in range(n):
        if labels[i] == -1:  # Not yet assigned
            # Start new cluster
            _flood_fill(A, labels, i, current_label)
            current_label += 1
    
    return labels


def _flood_fill(A: torch.Tensor, labels: torch.Tensor, start: int, label: int) -> None:
    """Flood fill algorithm to find connected components."""
    stack = [start]
    
    while stack:
        node = stack.pop()
        if labels[node] != -1:  # Already visited
            continue
            
        labels[node] = label
        
        # Add all unvisited neighbors to stack
        neighbors = torch.nonzero(A[node], as_tuple=True)[0]
        for neighbor in neighbors:
            if labels[neighbor] == -1:
                stack.append(int(neighbor.item()))
    # >>> END:AI_EDIT
