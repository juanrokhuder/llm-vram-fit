from model_fit_advisor import compute_weight_memory_gib, compute_artifact_size_gib, collect_gguf_artifacts

mock_repo_data = {
    "siblings": [
        {"rfilename": "model-Q4.gguf", "size": 4_294_967_296},
        {"rfilename": "README.md", "size": 100}
    ]
}

def test_compute_weight_memory_gib():
    assert compute_weight_memory_gib(1_073_741_824, 1) == 1

def test_compute_artifact_size_gib():
    assert compute_artifact_size_gib(4_294_967_296) == 4

def test_collect_gguf_artifacts():
    expect = [{"filename": "model-Q4.gguf", "file_size_bytes": 4_294_967_296, "artifact_size_gib": 4}]
    assert collect_gguf_artifacts(mock_repo_data) == expect