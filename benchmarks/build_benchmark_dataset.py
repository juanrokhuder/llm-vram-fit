from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from parse_llama_log import parse_llama_log

BENCHMARK_DIR = Path(__file__).resolve().parent

log_file_paths = (BENCHMARK_DIR / "raw").glob("*.log")
records = []

for file in log_file_paths:
    record = parse_llama_log(file)
    records.append(record)

data_frame = pd.DataFrame(records).sort_values(by="ubatch_size")

data_frame["tracked_cuda_buffers_mib"] = (
    data_frame["cuda_model_buffer_mib"]
    + data_frame["cuda_kv_buffer_mib"]
    + data_frame["cuda_compute_buffer_mib"]
)

ubatch_data_frame = data_frame[
    data_frame["context_length"] == 8192
].sort_values(by="ubatch_size")

context_data_frame = data_frame[
    data_frame["ubatch_size"] == 256
].sort_values(by="context_length")

data_frame.to_csv(BENCHMARK_DIR / "benchmark_results.csv", index=False)

print(
    ubatch_data_frame[
        [
            "ubatch_size",
            "cuda_compute_buffer_mib",
            "generation_tokens_per_second",
            "tracked_cuda_buffers_mib",
        ]
    ].to_string(index=False)
)

plt.plot(
    ubatch_data_frame["ubatch_size"],
    ubatch_data_frame["cuda_compute_buffer_mib"],
    marker="o",
)
plt.xlabel("Micro-batch size (tokens)")
plt.ylabel("CUDA compute buffer (MiB)")
plt.title("CUDA Compute Buffer Scaling by Micro-Batch Size")
plt.grid(True)
plt.tight_layout()
plt.savefig(BENCHMARK_DIR / "ubatch_compute_buffer.png")
plt.close()

print(
    context_data_frame[
        [
            "context_length",
            "cuda_kv_buffer_mib",
            "ubatch_size",
            "generation_tokens_per_second",
            "tracked_cuda_buffers_mib",
        ]
    ].to_string(index=False)
)

plt.plot(
    context_data_frame["context_length"],
    context_data_frame["cuda_kv_buffer_mib"],
    marker="o",
)
plt.xlabel("Context length (tokens)")
plt.ylabel("CUDA KV buffer (MiB)")
plt.title("CUDA KV Buffer Scaling by Context Length")
plt.grid(True)
plt.tight_layout()
plt.savefig(BENCHMARK_DIR / "context_kv_buffer.png")
plt.close()
