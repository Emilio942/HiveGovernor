import torch

from swarm_moi.clustering import cluster_from_C


def test_clustering_clear_blocks():
    """Test clustering with clear block structure."""
    # Create a co-activation matrix with two clear blocks
    C = torch.tensor([
        [1.0, 0.8, 0.1, 0.0],
        [0.8, 1.0, 0.1, 0.0],
        [0.1, 0.1, 1.0, 0.9],
        [0.0, 0.0, 0.9, 1.0],
    ], dtype=torch.float32)
    
    # Test threshold-based clustering
    labels_thresh = cluster_from_C(C, method="threshold", threshold=0.5)
    
    # Should have exactly 2 clusters
    unique_labels = torch.unique(labels_thresh)
    assert len(unique_labels) == 2, f"Expected 2 clusters, got {len(unique_labels)}"
    
    # Experts 0,1 should be in same cluster; experts 2,3 should be in same cluster
    assert labels_thresh[0] == labels_thresh[1], "Experts 0 and 1 should be in same cluster"
    assert labels_thresh[2] == labels_thresh[3], "Experts 2 and 3 should be in same cluster"
    assert labels_thresh[0] != labels_thresh[2], "Experts 0 and 2 should be in different clusters"
    
    # Test spectral clustering
    labels_spectral = cluster_from_C(C, method="spectral")
    
    # Should have 2 distinct values
    unique_spectral = torch.unique(labels_spectral)
    assert len(unique_spectral) == 2, f"Spectral clustering should find 2 clusters, got {len(unique_spectral)}"


def test_clustering_single_cluster():
    """Test clustering when all experts are strongly connected."""
    # All experts have strong co-activation
    C = torch.tensor([
        [1.0, 0.9, 0.8],
        [0.9, 1.0, 0.7],
        [0.8, 0.7, 1.0],
    ], dtype=torch.float32)
    
    labels = cluster_from_C(C, method="threshold", threshold=0.6)
    
    # All should be in the same cluster
    unique_labels = torch.unique(labels)
    assert len(unique_labels) == 1, f"Expected 1 cluster, got {len(unique_labels)}"


def test_clustering_isolated_experts():
    """Test clustering with some isolated experts."""
    # Expert 0 is isolated, experts 1,2 are connected
    C = torch.tensor([
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.8],
        [0.1, 0.8, 1.0],
    ], dtype=torch.float32)
    
    labels = cluster_from_C(C, method="threshold", threshold=0.5)
    
    # Should have 2 clusters: {0} and {1,2}
    unique_labels = torch.unique(labels)
    assert len(unique_labels) == 2, f"Expected 2 clusters, got {len(unique_labels)}"
    
    # Expert 0 should be alone
    assert labels[1] == labels[2], "Experts 1 and 2 should be in same cluster"
    assert labels[0] != labels[1], "Expert 0 should be in different cluster from experts 1,2"


def test_clustering_empty_matrix():
    """Test clustering with zero co-activations."""
    C = torch.zeros((3, 3), dtype=torch.float32)
    
    labels = cluster_from_C(C, method="threshold", threshold=0.1)
    
    # Each expert should be in its own cluster
    unique_labels = torch.unique(labels)
    assert len(unique_labels) == 3, f"Expected 3 clusters for isolated experts, got {len(unique_labels)}"


def test_clustering_method_validation():
    """Test that invalid methods raise appropriate errors."""
    C = torch.eye(3, dtype=torch.float32)
    
    # Invalid method should raise ValueError
    try:
        cluster_from_C(C, method="invalid_method")
        assert False, "Should have raised ValueError for invalid method"
    except ValueError as e:
        assert "Unknown clustering method" in str(e)
    
    # Invalid matrix shape should raise ValueError
    try:
        cluster_from_C(torch.randn(3, 4), method="threshold")
        assert False, "Should have raised ValueError for non-square matrix"
    except ValueError as e:
        assert "square matrix" in str(e)
