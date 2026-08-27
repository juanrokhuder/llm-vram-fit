import subprocess

from model_fit_advisor import (
    compute_artifact_size_gib,
    collect_gguf_artifacts,
    compute_memory_budget_gib,
    compute_kv_per_token_gib,
    compute_max_context,
    compute_usable_context,
    detect_nvidia_gpus,
    normalize_repo_id,
    extract_base_model_repo_id,
    fetch_model_config,
    build_model_entry,
)

mock_repo_data = {
    "siblings": [
        {"rfilename": "model-Q4.gguf", "size": 4_294_967_296},
        {"rfilename": "README.md", "size": 100},
    ]
}


def test_normalize_repo_id():
    assert normalize_repo_id("Qwen/Qwen3-8B-GGUF") == "Qwen/Qwen3-8B-GGUF"
    assert (
        normalize_repo_id(" https://huggingface.co/Qwen/Qwen3-8B-GGUF/ ")
        == "Qwen/Qwen3-8B-GGUF"
    )


def test_extract_base_model_repo_id():
    repo_data = {"cardData": {"base_model": "Qwen/Qwen3-8B"}}

    assert extract_base_model_repo_id(repo_data) == "Qwen/Qwen3-8B"

    repo_data = {"cardData": {"base_model": ["Qwen/Qwen3-8B"]}}

    assert extract_base_model_repo_id(repo_data) == "Qwen/Qwen3-8B"


def test_fetch_model_config(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"num_hidden_layers": 36}

    monkeypatch.setattr(
        "model_fit_advisor.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    assert fetch_model_config("Qwen/Qwen3-8B") == {"num_hidden_layers": 36}


def test_build_model_entry(monkeypatch):
    repo_data = {"cardData": {"base_model": "Qwen/Qwen3-8B"}}

    monkeypatch.setattr(
        "model_fit_advisor.fetch_repo_data",
        lambda repo_id: repo_data,
    )

    model_config = {
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "max_position_embeddings": 40960,
    }

    monkeypatch.setattr(
        "model_fit_advisor.fetch_model_config",
        lambda repo_id: model_config,
    )

    expected = {
        "name": "Qwen3-8B-GGUF",
        "repo_id": "Qwen/Qwen3-8B-GGUF",
        "num_hidden_layers": 36,
        "num_key_and_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
    }

    assert build_model_entry("https://huggingface.co/Qwen/Qwen3-8B-GGUF/") == expected


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


def test_detect_nvidia_gpus(monkeypatch):
    fake_result = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="GPU A, 4096, 3072\nGPU B, 8192, 6144\n",
        stderr="",
    )
    monkeypatch.setattr(
        "model_fit_advisor.subprocess.run",
        lambda *args, **kwargs: fake_result,
    )
    expected = [
        {
            "name": "GPU A",
            "total_memory_mib": 4096,
            "free_memory_mib": 3072,
        },
        {
            "name": "GPU B",
            "total_memory_mib": 8192,
            "free_memory_mib": 6144,
        },
    ]

    assert detect_nvidia_gpus() == expected


def test_detect_nvidia_gpus_returns_empty_when_command_is_missing(monkeypatch):
    def raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(
        "model_fit_advisor.subprocess.run",
        raise_file_not_found,
    )

    assert detect_nvidia_gpus() == []
