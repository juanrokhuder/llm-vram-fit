# llm-vram-fit

This tool checks if any chosen AI Model fits inside your GPU at different weight precisions.

## Usage

Run `python3 model_fit_advisor.py` and then submit your GPU size in GiB or skip the prompt by adding the GPU size number after file name, for example: `python3 model_fit_advisor.py 32`.

## Adding models

To add/change/remove models open `models.json` and update it with your desired models. To find the config values of your model(s) visit https://huggingface.co/models, search for your model and read its `config.json`. You will need those 3 values: `num_hidden_layers`, `num_key_value_heads`, `head_dim`. If you can't find `head_dim` take `hidden_size` divided by `num_attention_heads`.

## Assumptions

This tool takes certain numbers for granted. In reality values differ and fluctuate:

- CUDA overhead is assumed at 0.5 GiB
- KV cache is always fp16 regardless of weight precision
- int4 is treated as 0.5 bytes per parameter while real quantized files measure closer to 0.63

## Example output

```

python3 model_fit_advisor.py

What is the size of your GPU in GiB?: 32

This is the report for the model: Qwen3-8B:
The total weight memory with fp16 weight precision is 15.27 GiB.
The total memory left over for KV cache with fp16 weight precision is 16.23 GiB.
The maximum context length that fits with fp16 weight precision is 118156.
The model with fp16 weight precision fits inside your 32.0 GiB GPU.
The total weight memory with int8 weight precision is 7.64 GiB.
The total memory left over for KV cache with int8 weight precision is 23.86 GiB.
The maximum context length that fits with int8 weight precision is 173766.
The model with int8 weight precision fits inside your 32.0 GiB GPU.
The total weight memory with int4 weight precision is 3.82 GiB.
The total memory left over for KV cache with int4 weight precision is 27.68 GiB.
The maximum context length that fits with int4 weight precision is 201571.
The model with int4 weight precision fits inside your 32.0 GiB GPU.


```
