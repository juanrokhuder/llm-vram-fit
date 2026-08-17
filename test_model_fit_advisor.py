from model_fit_advisor import (
    compute_artifact_size_gib,
    collect_gguf_artifacts,
    compute_memory_budget_gib,
    compute_kv_per_token_gib,
    compute_max_context,
    compute_usable_context,
)

mock_repo_data = {
    "siblings": [
        {"rfilename": "model-Q4.gguf", "size": 4_294_967_296},
        {"rfilename": "README.md", "size": 100},
    ]
}


def test_compute_artifact_size_gib():
    assert compute_artifact_size_gib(4_294_967_296) == 4


def test_collect_gguf_artifacts():
    expected = [
        {
            "filename": "model-Q4.gguf",
            "file_size_bytes": 4_294_967_296,
            "artifact_size_gib": 4,
        }
    ]
    assert collect_gguf_artifacts(mock_repo_data) == expected


def test_compute_memory_budget_gib():
    assert compute_memory_budget_gib(8, 5.5) == 2


def test_compute_kv_per_token_gib():
    assert compute_kv_per_token_gib(32, 8, 128) == 0.0001220703125


def test_compute_max_context():
    assert compute_max_context(1, 0.0001220703125) == 8192


def test_usable_context_uses_model_limit():
    assert compute_usable_context(420, 69) == 69


def test_usable_context_uses_memory_limit():
    assert compute_usable_context(21, 80085) == 21
