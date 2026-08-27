import argparse
import json
import subprocess
from pathlib import Path

import requests

HF_API_BASE_URL = "https://huggingface.co/api/models/"
REQUEST_TIMEOUT_SECONDS = 30

BYTES_PER_GIB = 1024**3

# Reserve an estimated amount for CUDA runtime overhead not included in the GGUF artifact size.
CUDA_OVERHEAD_GIB = 0.5

KEY_AND_VALUE_TENSORS = 2
# Assumes FP16 KV-cache storage at two bytes per element.
BYTES_PER_KV_CACHE_ELEMENT = 2

# Resolve models.json relative to this script rather than the current working directory.
MODELS_PATH = Path(__file__).with_name("models.json")


def normalize_repo_id(repo_input):
    """Return a normalized Hugging Face repository ID."""
    repo_input = repo_input.strip()
    prefix = "https://huggingface.co/"
    if repo_input.startswith(prefix):
        repo_input = repo_input[len(prefix) :]

    return repo_input.rstrip("/")


def extract_base_model_repo_id(repo_data):
    """Return the base-model repository ID declared in repository metadata."""
    base_model = repo_data.get("cardData", {}).get("base_model")
    if isinstance(base_model, list):
        base_model = base_model[0] if base_model else None
    if not base_model:
        raise ValueError("The GGUF repository does not identify a base model.")

    return normalize_repo_id(base_model)


def fetch_repo_data(repo_id):
    """Fetch and return metadata for a Hugging Face model repository."""
    response = requests.get(
        f"{HF_API_BASE_URL}{repo_id}?blobs=true", timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


def fetch_model_config(repo_id):
    """Fetch and return a Hugging Face model repository's config.json."""
    response = requests.get(
        f"https://huggingface.co/{repo_id}/resolve/main/config.json",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    return response.json()


def build_model_entry(repo_input):
    """Build a model configuration entry from a GGUF repository ID or URL."""
    repo_id = normalize_repo_id(repo_input)
    repo_data = fetch_repo_data(repo_id)
    base_model_repo_id = extract_base_model_repo_id(repo_data)
    model_config = fetch_model_config(base_model_repo_id)
    head_dim = model_config.get("head_dim")
    if head_dim is None:
        head_dim = model_config["hidden_size"] // model_config["num_attention_heads"]

    return {
        "name": repo_id.rsplit("/", 1)[-1],
        "repo_id": repo_id,
        "num_hidden_layers": model_config["num_hidden_layers"],
        "num_key_and_value_heads": model_config.get(
            "num_key_value_heads", model_config["num_attention_heads"]
        ),
        "head_dim": head_dim,
        "max_position_embeddings": model_config["max_position_embeddings"],
    }


def compute_artifact_size_gib(file_size_bytes):
    """Convert an artifact size from bytes to GiB."""
    total_artifact_gib = file_size_bytes / BYTES_PER_GIB
    return total_artifact_gib


def collect_gguf_artifacts(repo_data):
    """Extract GGUF artifact filenames and byte/GiB sizes from repository metadata."""
    data = []
    for file in repo_data["siblings"]:
        if file["rfilename"].endswith(".gguf"):
            file_size_bytes = file["size"]
            data.append(
                {
                    "filename": file["rfilename"],
                    "file_size_bytes": file_size_bytes,
                    "artifact_size_gib": compute_artifact_size_gib(file_size_bytes),
                }
            )
    return data


def compute_memory_budget_gib(gpu_size_gib, artifact_size_gib):
    """Return VRAM available for KV cache after the artifact and estimated CUDA overhead."""
    total_memory_budget_gib = gpu_size_gib - artifact_size_gib - CUDA_OVERHEAD_GIB
    return total_memory_budget_gib


def compute_kv_per_token_gib(layers, kv_heads, head_dim):
    """Estimate KV cache memory in GiB per token for a model architecture."""
    total_kv_per_token_gib = (
        KEY_AND_VALUE_TENSORS
        * layers
        * kv_heads
        * head_dim
        * BYTES_PER_KV_CACHE_ELEMENT
    ) / BYTES_PER_GIB
    return total_kv_per_token_gib


def compute_max_context(memory_budget_gib, kv_per_token_gib):
    """Return the number of KV cache tokens that fit within the memory budget."""
    total_tokens = memory_budget_gib / kv_per_token_gib
    return int(total_tokens)


def compute_usable_context(memory_limited_context, model_supported_context):
    """Return the smaller value of the memory limited and model supported context lengths."""
    usable_context = min(memory_limited_context, model_supported_context)
    return usable_context


def detect_nvidia_gpus():
    nvidia_gpus = []
    try:
        results = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return nvidia_gpus

    for line in results.stdout.splitlines():
        values = line.split(",")
        name = values[0].strip()
        total_memory_mib = values[1].strip()
        free_memory_mib = values[2].strip()
        gpu = {
            "name": name,
            "total_memory_mib": int(total_memory_mib),
            "free_memory_mib": int(free_memory_mib),
        }
        nvidia_gpus.append(gpu)
    return nvidia_gpus


def main():
    """Run the CLI and generate model fit reports."""

    parser = argparse.ArgumentParser(
        description="Estimate whether GGUF model artifacts fit in available GPU VRAM."
    )

    parser.add_argument(
        "gpu_size_gib",
        nargs="?",
        type=float,
        help="Manual available VRAM override in GiB",
    )

    parser.add_argument(
        "--repo",
        help="Hugging Face GGUF repository or URL",
    )

    args = parser.parse_args()

    if args.repo is not None:
        models = [build_model_entry(args.repo)]
    else:
        with open(MODELS_PATH, "r") as f:
            models = json.load(f)

    gpu_targets = []
    # Use a positional CLI argument when provided, otherwise prompt interactively.
    if args.gpu_size_gib is not None:
        gpu_targets.append(
            {
                "name": "Manual override",
                "gpu_memory_gib": args.gpu_size_gib,
            }
        )
    else:
        detected_gpus = detect_nvidia_gpus()
        if detected_gpus:
            for gpu in detected_gpus:
                gpu_memory_gib = gpu["free_memory_mib"] / 1024
                gpu_targets.append(
                    {
                        "name": gpu["name"],
                        "gpu_memory_gib": gpu_memory_gib,
                    }
                )
        else:
            while True:
                input_gpu_size_gib = input("What is the size of your GPU in GiB?: ")
                try:
                    gpu_memory_gib = float(input_gpu_size_gib)
                    gpu_targets.append(
                        {
                            "name": "Manual input",
                            "gpu_memory_gib": gpu_memory_gib,
                        }
                    )
                    break
                except ValueError:
                    print("\nERROR: Please submit only a number.\n")

    reports = []
    for gpu_target in gpu_targets:
        gpu_size_gib = gpu_target["gpu_memory_gib"]
        gpu_report_header = (
            f"GPU: {gpu_target['name']} ({gpu_size_gib:.2f} GiB available)\n\n"
        )
        reports.append(gpu_report_header)
        print(gpu_report_header, end="")

        for config in models:
            model_report_parts = [f"This is the report for {config['name']}.\n"]

            kv_per_token_gib = compute_kv_per_token_gib(
                config["num_hidden_layers"],
                config["num_key_and_value_heads"],
                config["head_dim"],
            )

            repo_data = fetch_repo_data(config["repo_id"])

            model_supported_context = config["max_position_embeddings"]

            gguf_artifacts = collect_gguf_artifacts(repo_data)

            # Evaluate each GGUF artifact independently against the available VRAM.
            for artifact in gguf_artifacts:
                filename = artifact["filename"]
                size_gib = artifact["artifact_size_gib"]
                memory_budget_gib = compute_memory_budget_gib(gpu_size_gib, size_gib)
                if memory_budget_gib > 0:
                    memory_message = (
                        "After the GGUF artifact and estimated CUDA overhead, "
                        f"{memory_budget_gib:.2f} GiB remains.\n"
                    )
                    memory_limited_context = compute_max_context(
                        memory_budget_gib, kv_per_token_gib
                    )
                    usable_context = compute_usable_context(
                        memory_limited_context, model_supported_context
                    )
                    context_message = (
                        f"The maximum usable context for {filename} is "
                        f"{usable_context} tokens.\n"
                    )
                else:
                    memory_deficit_gib = abs(memory_budget_gib)
                    memory_message = (
                        f"The GGUF artifact and estimated CUDA overhead exceed the available VRAM "
                        f"by {memory_deficit_gib:.2f} GiB.\n"
                    )
                    context_message = (
                        f"{filename} does not fit within the available VRAM.\n"
                    )

                entry = (
                    f"These were the findings for {filename} with the size {size_gib:.2f} GiB:\n"
                    + memory_message
                    + context_message
                )

                model_report_parts.append(entry)

            model_report = "".join(model_report_parts)
            reports.append(model_report)
            print(model_report)

    # Write the combined report to the caller's working directory.
    with open("report.txt", "w") as f:
        f.write("\n".join(reports))


if __name__ == "__main__":
    main()
