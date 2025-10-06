import torch

from swarm_moi.reporter import build_report


def test_reporter_schema_keys():
    """Test that reporter returns all expected keys."""
    # Create minimal inputs
    C = torch.eye(3, dtype=torch.float32)
    loop_pairs = [(0, 1, 0.5)]
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    
    report = build_report(C, loop_pairs, labels)
    
    # Check all required keys are present
    expected_keys = {'top_groups', 'top_loop_pairs', 'inhibition_history', 'summary'}
    assert set(report.keys()) == expected_keys, f"Missing keys: {expected_keys - set(report.keys())}"
    
    # Check top_groups structure
    assert isinstance(report['top_groups'], list)
    if report['top_groups']:
        group = report['top_groups'][0]
        group_keys = {'label', 'size', 'experts', 'representative', 'avg_coactivation'}
        assert set(group.keys()) == group_keys, f"Group missing keys: {group_keys - set(group.keys())}"
    
    # Check top_loop_pairs structure
    assert isinstance(report['top_loop_pairs'], list)
    if report['top_loop_pairs']:
        pair = report['top_loop_pairs'][0]
        pair_keys = {'expert1', 'expert2', 'strength', 'pair_id'}
        assert set(pair.keys()) == pair_keys, f"Pair missing keys: {pair_keys - set(pair.keys())}"
    
    # Check inhibition_history structure
    assert isinstance(report['inhibition_history'], dict)
    history_keys = {'total_inhibitions', 'recent_inhibitions', 'most_inhibited_experts', 'note'}
    hist = report['inhibition_history']
    assert set(hist.keys()) == history_keys, f"History missing keys: {history_keys - set(hist.keys())}"
    
    # Check summary structure
    assert isinstance(report['summary'], dict)
    summary_keys = {'n_experts', 'total_groups', 'largest_group_size', 'total_loop_pairs', 
                   'strongest_loop_strength', 'fragmentation_ratio'}
    summary = report['summary']
    assert set(summary.keys()) == summary_keys, f"Summary missing keys: {summary_keys - set(summary.keys())}"


def test_reporter_values_reasonable():
    """Test that reporter produces reasonable values."""
    # Create test data with clear structure
    C = torch.tensor([
        [1.0, 0.8, 0.1, 0.0],
        [0.8, 1.0, 0.1, 0.0],
        [0.1, 0.1, 1.0, 0.9],
        [0.0, 0.0, 0.9, 1.0],
    ], dtype=torch.float32)
    
    loop_pairs = [(0, 1, 0.8), (2, 3, 0.9)]
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Two groups: {0,1} and {2,3}
    
    report = build_report(C, loop_pairs, labels)
    
    # Check summary values
    summary = report['summary']
    assert summary['n_experts'] == 4
    assert summary['total_groups'] == 2
    assert summary['largest_group_size'] == 2
    assert summary['total_loop_pairs'] == 2
    assert summary['strongest_loop_strength'] == 0.8  # First pair in list
    assert 0.0 <= summary['fragmentation_ratio'] <= 1.0
    
    # Check groups
    groups = report['top_groups']
    assert len(groups) == 2
    # Groups should be sorted by size (both have size 2, so order may vary)
    for group in groups:
        assert group['size'] == 2
        assert len(group['experts']) == 2
        assert group['representative'] in group['experts']
        assert group['avg_coactivation'] >= 0.0
    
    # Check loop pairs
    pairs = report['top_loop_pairs']
    assert len(pairs) == 2
    assert pairs[0]['strength'] == 0.8
    assert pairs[1]['strength'] == 0.9
    assert pairs[0]['pair_id'] == "0-1"
    assert pairs[1]['pair_id'] == "2-3"


def test_reporter_empty_inputs():
    """Test reporter with empty or minimal inputs."""
    # Empty inputs
    C_empty = torch.empty(0, 0, dtype=torch.float32)
    loop_pairs_empty = []
    labels_empty = torch.empty(0, dtype=torch.long)
    
    report = build_report(C_empty, loop_pairs_empty, labels_empty)
    
    assert report['summary']['n_experts'] == 0
    assert report['summary']['total_groups'] == 0
    assert report['summary']['total_loop_pairs'] == 0
    assert len(report['top_groups']) == 0
    assert len(report['top_loop_pairs']) == 0
    
    # Single expert
    C_single = torch.tensor([[1.0]], dtype=torch.float32)
    labels_single = torch.tensor([0], dtype=torch.long)
    
    report_single = build_report(C_single, [], labels_single)
    
    assert report_single['summary']['n_experts'] == 1
    assert report_single['summary']['total_groups'] == 1
    assert report_single['summary']['largest_group_size'] == 1
    assert len(report_single['top_groups']) == 1
    assert report_single['top_groups'][0]['experts'] == [0]


def test_reporter_max_limits():
    """Test that reporter respects max_groups and max_loop_pairs limits."""
    # Create many groups and pairs
    C = torch.eye(10, dtype=torch.float32)
    loop_pairs = [(i, i+1, 0.5 + i*0.1) for i in range(8)]  # 8 pairs
    labels = torch.arange(10, dtype=torch.long)  # 10 individual groups
    
    # Test with small limits
    report = build_report(C, loop_pairs, labels, max_groups=3, max_loop_pairs=2)
    
    assert len(report['top_groups']) <= 3
    assert len(report['top_loop_pairs']) <= 2
    assert report['summary']['total_groups'] == len(report['top_groups'])
    assert report['summary']['total_loop_pairs'] == len(report['top_loop_pairs'])


def test_reporter_group_analysis():
    """Test detailed group analysis functionality."""
    # Create a scenario with groups of different sizes
    C = torch.zeros(6, 6, dtype=torch.float32)
    # Group 0: experts 0,1,2 (size 3)
    C[0:3, 0:3] = 0.7
    # Group 1: experts 3,4 (size 2)  
    C[3:5, 3:5] = 0.8
    # Group 2: expert 5 (size 1)
    C[5, 5] = 1.0
    
    labels = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    
    report = build_report(C, [], labels)
    
    groups = report['top_groups']
    # Should be sorted by size: [group0(size=3), group1(size=2), group2(size=1)]
    assert groups[0]['size'] == 3
    assert groups[1]['size'] == 2
    assert groups[2]['size'] == 1
    
    # Check that representatives are valid
    for group in groups:
        assert group['representative'] in group['experts']
        assert group['avg_coactivation'] >= 0.0
