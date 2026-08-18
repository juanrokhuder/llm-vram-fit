from pathlib import Path
import pandas as pd

from parse_llama_log import parse_llama_log
import matplotlib.pyplot as plt

log_file_paths = Path("raw").glob("*.log")
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

data_frame.to_csv("benchmark_results.csv", index=False)

print(
    data_frame[
        [
            "ubatch_size",
            "cuda_compute_buffer_mib",
            "generation_tokens_per_second",
            "tracked_cuda_buffers_mib",
        ]
    ].to_string(index=False)
)

plt.plot(
    data_frame["ubatch_size"],
    data_frame["cuda_compute_buffer_mib"],
    marker="o",
)
plt.xlabel("Micro-batch size (tokens)")
plt.ylabel("CUDA compute buffer (MiB)")
plt.title("CUDA Compute Buffer Scaling by Micro-Batch Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("ubatch_compute_buffer.png")


