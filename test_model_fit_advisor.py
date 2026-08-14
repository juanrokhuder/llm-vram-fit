from model_fit_advisor import compute_weight_memory_gib, compute_artifact_size_gib

def test_compute_weight_memory_gib():
    assert compute_weight_memory_gib(1_073_741_824, 1) == 1

def test_compute_artifact_size_gib():
    assert compute_artifact_size_gib(4_294_967_296) == 4