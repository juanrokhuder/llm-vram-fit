import json
import sys

# Estimated CUDA overhead in GiB as a constant
CUDA_OVERHEAD_GIB = 0.5

# fp16, int8 and int4 constant (bytes per parameter)
PRECISIONS = {"fp16": 2, "int8": 1, "int4": 0.5}


# KV constants
KEY_AND_VALUE = 2
KV_CACHE_BYTES = 2

# Function that computes the weight memory in GiB from given parameters and precision weight bytes per parameter
def compute_weight_memory_gib(parameter_count, bytes_per_parameter):
    total_gib = (parameter_count * bytes_per_parameter) / 1024**3
    return total_gib

# Function that computes the memory budget in GiB that is left for the KV cache
def compute_memory_budget_gib(gpu_size, weight_memory):
    total_gib = gpu_size - weight_memory - CUDA_OVERHEAD_GIB
    return total_gib

# Function that computes the total GiB KV costs per token
def compute_kv_per_token_gib(layers, kv_heads, head_dim):
    total_gib = (
        KEY_AND_VALUE * layers * kv_heads * head_dim * KV_CACHE_BYTES
    ) / 1024**3
    return total_gib

# Function that computes the maximum possible context length with given model and precision
def compute_max_context(memory_budget, kv_per_token):
    total_tokens = memory_budget / kv_per_token
    return total_tokens


# Reading the json file with the models and storing the data in a variable called "models"
with open("models.json", "r") as f:
    models = json.load(f)

# CLI option to type in GPU size in GiB
if len(sys.argv) > 1:
    try:
        gpu_size_gib = float(sys.argv[1])
    except ValueError:
        print(f"{sys.argv[1]} is not a number.")
        exit()
else:
    # Let the user try until he types in a float value
    while True:
        # Asking the user his GPU Size in GiB
        input_gpu_size_gib = input("What is the size of your GPU in GiB?: ")
        # Convert the string input to a float, if he doesn't submit a number, throw error
        try:
            gpu_size_gib = float(input_gpu_size_gib)
            break
        except ValueError:
            print("\nERROR: Please submit only a number.\n")

# List that collects all the reports, which will be written to report.txt
reports = []

# Looping over each model in models.json
for config in models:
    # List that collects each model report with a preset header line, gets overwritten each run
    lines = [f"This is the report for the model: {config["name"]}:\n"]

    # Assign the computed KV cache per token to a variable
    kv_per_token_gib = compute_kv_per_token_gib(
        config["num_hidden_layers"],
        config["num_key_and_value_heads"],
        config["head_dim"],
    )

    # Looping over the 3 weight precisions
    for precision, bytes_per_parameter in PRECISIONS.items():
        # Computing the weight memory of the current model & weight precision and assign it to a variable
        weight_memory_gib = compute_weight_memory_gib(
            config["parameter_count"], bytes_per_parameter
        )

        # Computing the memory budget of the current model & weight precision and assign it to a variable
        memory_budget_gib = compute_memory_budget_gib(gpu_size_gib, weight_memory_gib)

        # Write detailed a report block if the model fits inside user's GPU, otherwise tell the user his GPU is too small and where the issue is
        if memory_budget_gib > 0:
            max_context = compute_max_context(memory_budget_gib, kv_per_token_gib)
            block = (
                f"The total weight memory with {precision} weight precision is {weight_memory_gib:.2f} GiB.\n"
                f"The total memory left over for KV cache with {precision} weight precision is {memory_budget_gib:.2f} GiB.\n"
                f"The maximum context length that fits with {precision} weight precision is {int(max_context)}.\n"
                f"The model with {precision} weight precision fits inside your {gpu_size_gib} GiB GPU.\n"
            )
        else:
            block = (
                f"The total weight memory with {precision} weight precision is {weight_memory_gib:.2f} GiB.\n"
                f"The model with {precision} weight precision doesn't fit inside your {gpu_size_gib} GiB GPU.\n"
            )

        # Appending the written block to the previous created model report list
        lines.append(block)

    # Converting the list to a string and assign it to a variable (report), then append the string to the reports list that collects all the reports in one list, then print the report of the current model run, then append
    report = "".join(lines)
    reports.append(report)
    print(report)

# Write the list that holds all the model reports into a report.txt file as a string
with open("report.txt", "w") as f:
    f.write("\n".join(reports))
