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
    """Simple spectral clustering using eigenvectors of the normalized Laplacian."""
    n = C.size(0)
    
    if n <= 1:
        return torch.zeros(n, dtype=torch.long)
    
    # Create adjacency matrix from co-activation (use absolute values)
    A = torch.abs(C)
    
    # Remove self-loops for cleaner clustering
    A = A - torch.diag(torch.diag(A))
    
    # Compute degree matrix
    d = A.sum(dim=1)
    
    # Handle disconnected components
    isolated = (d == 0)
    if isolated.all():
        # All nodes are isolated
        return torch.arange(n, dtype=torch.long)
    
    # For connected components, add small epsilon to avoid numerical issues
    d = torch.clamp(d, min=1e-6)
    D_sqrt_inv = torch.diag(1.0 / torch.sqrt(d))
    
    # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
    L_norm = torch.eye(n, device=C.device, dtype=C.dtype) - D_sqrt_inv @ A @ D_sqrt_inv
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvals, eigenvecs = torch.linalg.eigh(L_norm)
    except RuntimeError:
        # Fallback to threshold clustering if spectral fails
        return _threshold_clustering(C, threshold=0.5)
    
    # Use the Fiedler vector (second smallest eigenvalue) for binary clustering
    if n < 2:
        return torch.zeros(n, dtype=torch.long)
    
    fiedler_vec = eigenvecs[:, 1]  # eigenvalues are sorted in ascending order
    
    # Simple threshold-based assignment using sign or median
    # Use sign if the vector has both positive and negative values
    labels: torch.Tensor
    if (fiedler_vec > 0).any() and (fiedler_vec < 0).any():
        labels = (fiedler_vec >= 0).long()
    else:
        # Use median split if all values have same sign
        median_val = torch.median(fiedler_vec).values
        labels = (fiedler_vec >= median_val).long()
    
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
