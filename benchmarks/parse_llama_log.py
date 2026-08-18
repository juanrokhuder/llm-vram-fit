import sys

def parse_llama_log(log_file_path):
    """Parse a llama.cpp log file into structured benchmark metrics"""

    with open(log_file_path, "r") as f:
        log_file = f.read()

    for line in log_file.splitlines():
        if "common_param:" in line and "- CUDA0" in line:
            gpu_details = line.split("CUDA0", 1)[1]
            gpu_details = gpu_details.split(":", 1)[1].strip()

            gpu_name = gpu_details.split("(", 1)[0].strip()

            gpu_memory = gpu_details.split("(", 1)[1]
            gpu_total_memory_mib = int(gpu_memory.split("MiB", 1)[0].strip())
            gpu_free_memory_mib = gpu_memory.split(",", 1)[1]
            gpu_free_memory_mib = int(gpu_free_memory_mib.split("MiB", 1)[0].strip())
        if "load_model: loading model '" in line:
            model_filename = line.split("loading model '", 1)[1]
            model_filename = model_filename.split("'", 1)[0]
        if "print_info: file type" in line:
            quantization = line.split("=", 1)[1].strip()
        if "n_ubatch" in line:
            ubatch_size = int(line.split("=", 1)[1].strip())
        if "CUDA0 compute buffer size" in line:
            cuda_compute_buffer_mib = line.split("=", 1)[1]
            cuda_compute_buffer_mib = float(cuda_compute_buffer_mib.split("MiB", 1)[0].strip())
        key = line.split("=", 1)[0].strip()
        if key.endswith("llama_context: n_ctx"):
            context_length = int(line.split("=", 1)[1].strip())
        if "CUDA0 model buffer size" in line:
            cuda_model_buffer_mib = line.split("=", 1)[1]
            cuda_model_buffer_mib = float(cuda_model_buffer_mib.split("MiB", 1)[0].strip())
        if "CUDA0 KV buffer size" in line:
            cuda_kv_buffer_mib = line.split("=", 1)[1]
            cuda_kv_buffer_mib = float(cuda_kv_buffer_mib.split("MiB", 1)[0].strip())
        if "CUDA_Host" in line and "output buffer size" in line:
            host_output_buffer_mib = line.split("=", 1)[1]
            host_output_buffer_mib = float(host_output_buffer_mib.split("MiB", 1)[0].strip())
        if "CUDA_Host compute buffer size" in line:
            host_compute_buffer_mib = line.split("=", 1)[1]
            host_compute_buffer_mib = float(host_compute_buffer_mib.split("MiB", 1)[0].strip())
        if "prompt eval time" in line:
            prompt_eval_time_ms = line.split("=", 1)[1]
            prompt_eval_time_ms = float(prompt_eval_time_ms.split("ms", 1)[0].strip())

            prompt_tokens = line.split("/", 1)[1]
            prompt_tokens = int(prompt_tokens.split("tokens", 1)[0].strip())

            prompt_tokens_per_second = line.split(",", 1)[1]
            prompt_tokens_per_second = float(
                prompt_tokens_per_second.split("tokens per second", 1)[0].strip()
            )
        if "eval time" in line and "prompt eval time" not in line:
            generation_time_ms = line.split("=", 1)[1]
            generation_time_ms = float(generation_time_ms.split("ms", 1)[0].strip())

            generated_tokens = line.split("/", 1)[1]
            generated_tokens = int(generated_tokens.split("tokens", 1)[0].strip())

            generation_tokens_per_second = line.split(",", 1)[1]
            generation_tokens_per_second = float(
                generation_tokens_per_second.split("tokens per second", 1)[0].strip()
            )

    log_values = {
        "model_filename": model_filename,
        "quantization": quantization,
        "gpu_name": gpu_name,
        "gpu_total_memory_mib": gpu_total_memory_mib,
        "gpu_free_memory_mib": gpu_free_memory_mib,
        "ubatch_size": ubatch_size,
        "cuda_compute_buffer_mib": cuda_compute_buffer_mib,
        "context_length": context_length,
        "cuda_model_buffer_mib": cuda_model_buffer_mib,
        "cuda_kv_buffer_mib": cuda_kv_buffer_mib,
        "host_output_buffer_mib": host_output_buffer_mib,
        "host_compute_buffer_mib": host_compute_buffer_mib,
        "prompt_eval_time_ms": prompt_eval_time_ms,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_second": prompt_tokens_per_second,
        "generation_time_ms": generation_time_ms,
        "generated_tokens": generated_tokens,
        "generation_tokens_per_second": generation_tokens_per_second,
    }

    return log_values 


if __name__ == "__main__":

    log_file_path = sys.argv[1]
    print(parse_llama_log(log_file_path))