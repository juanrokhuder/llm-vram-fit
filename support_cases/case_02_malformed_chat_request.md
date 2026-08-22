# Case 02: Malformed chat completion request

## Customer report

My `curl` request to `v1/chat/completions` fails after I add `/no_think`. The server returns a JSON parsing error, and `curl` also reports that part of the command is an invalid URL. I need to understand why the request is being split incorrectly and how to send it successfully.

## Observed behavior

```bash
curl.exe http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-0.6B-Q8_0.gguf","messages":[{"role":"user","content":"In one sentence, explain GPU inference. /no_think"}], "max_tokens":64}
```

After Enter, the terminal displayed a `>` continuation prompt. Pasting the command again then produced:

```text
JSON parse error: expected end of input
curl: (3) bad range
```

## Initial interpretation

This error proves that the customer's command is unfinished judging by the `>`. The parse error also shows that the server received an invalid JSON and curl interpreted part of the malformed command as another argument or URL.

## First hypothesis

The customer might have a shell syntax mistake such as missing a bracket or a missing quote.

## Hypothesis check

Later, after running:

```bash
curl.exe http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-0.6B-Q8_0.gguf","messages":[{"role":"user","content":"In one sentence, explain GPU inference. /no_think"}], "max_tokens":64}'
```

the server returned a valid chat-completion response.

## Root cause

After the last curly bracket there is a missing single quote that was opened after -d but never closed anywhere in the command.

## Resolution

Add the missing single quote after the last curly bracket.

## Verification

```bash
curl.exe http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-0.6B-Q8_0.gguf","messages":[{"role":"user","content":"In one sentence, explain GPU inference. /no_think"}], "max_tokens":64}'
```

The server returned a successful chat-completion response (formatted and abridged for readability):

```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "GPU inference refers to the process of using a GPU to perform computer vision and deep learning tasks, enabling faster computation and processing of complex models."
      }
    }
  ]
}
```

## Prevention

Always ensure that the single quote opened after -d is closed after the JSON. If in the terminal a `>` appears unexpectedly press `Ctrl + C` and inspect the command instead of pasting it again.

## Customer response

We noticed the missing closing single quote in your command after the last curly bracket. That caused the shell to treat the command as unfinished and display the `>` continuation prompt. The corrected command goes as follows:

```bash
curl.exe http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-0.6B-Q8_0.gguf","messages":[{"role":"user","content":"In one sentence, explain GPU inference. /no_think"}], "max_tokens":64}'
```

After successfully adding the missing single quote, the server returned a valid chat-completion response.