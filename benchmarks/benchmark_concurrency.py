import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

LLAMA_SERVER_URL = os.environ.get(
    "LLAMA_SERVER_URL",
    "http://127.0.0.1:8080/v1/chat/completions",
)

BENCHMARK_DIR = Path(__file__).resolve().parent

PAYLOAD = {
    "model": "Qwen3-0.6B-Q8_0.gguf",
    "messages": [
        {"role": "user", "content": "In one sentence, explain GPU inference. /no_think"}
    ],
    "max_tokens": 64,
    "temperature": 0,
    "ignore_eos": True,
}

REQUEST_TIMEOUT_SECONDS = 120

def run_request():
    start = time.perf_counter()

    response = requests.post(
        LLAMA_SERVER_URL,
        json=PAYLOAD,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    end = time.perf_counter()
    response_data = response.json()

    end_to_end_latency = end - start
    predicted_per_second = response_data["timings"]["predicted_per_second"]
    completion_token_count = response_data["usage"]["completion_tokens"]
    response_content = response_data["choices"][0]["message"]["content"]

    return (
        end_to_end_latency,
        predicted_per_second,
        completion_token_count,
        response_content,
    )


def run_concurrency_test(concurrency, repetition):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_request) for _ in range(concurrency)]
        concurrent_results = [future.result() for future in futures]
    end = time.perf_counter()

    concurrent_duration = end - start
    total_concurrent_tokens = sum(result[2] for result in concurrent_results)
    aggregate_throughput = total_concurrent_tokens / concurrent_duration
    average_request_latency = (
        sum(result[0] for result in concurrent_results) / concurrency
    )
    maximum_request_latency = max(result[0] for result in concurrent_results)
    return {
        "concurrency": concurrency,
        "aggregate_throughput_tps": aggregate_throughput,
        "average_request_latency_s": average_request_latency,
        "maximum_request_latency_s": maximum_request_latency,
        "repetition": repetition,
    }

def main():
    for _ in range(2):
        run_request()

    repetitions = 5
    benchmark_results = []
    concurrencies = [1, 2, 3, 4, 5, 6, 7, 8]

    for repetition in range(repetitions):
        for concurrency in concurrencies:
            result = run_concurrency_test(concurrency, repetition + 1)
            benchmark_results.append(result)

    data_frame = pd.DataFrame(benchmark_results)
    data_frame.to_csv(BENCHMARK_DIR / "concurrency_benchmark_raw.csv", index=False)

    summary_data_frame = data_frame.groupby("concurrency", as_index=False)[
        [
            "aggregate_throughput_tps",
            "average_request_latency_s",
            "maximum_request_latency_s",
        ]
    ].mean()

    summary_data_frame.to_csv(
        BENCHMARK_DIR / "concurrency_benchmark_summary.csv", index=False
    )

    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(
        summary_data_frame["concurrency"],
        summary_data_frame["aggregate_throughput_tps"],
        marker="o",
    )
    axes[0].set_ylabel("Aggregate throughput (tokens/s)")
    axes[0].set_title("llama.cpp Concurrency Scaling")
    axes[0].grid(True)

    axes[1].plot(
        summary_data_frame["concurrency"],
        summary_data_frame["average_request_latency_s"],
        marker="o",
        label="Average latency",
    )
    axes[1].plot(
        summary_data_frame["concurrency"],
        summary_data_frame["maximum_request_latency_s"],
        marker="o",
        label="Maximum latency",
    )
    axes[1].set_xlabel("Concurrent requests")
    axes[1].set_ylabel("Latency (seconds)")
    axes[1].legend()
    axes[1].grid(True)

    figure.tight_layout()
    figure.savefig(BENCHMARK_DIR / "concurrency_scaling.png", dpi=150)
    plt.close(figure)

    print(summary_data_frame.to_string(index=False))


if __name__ == "__main__":
    main()
