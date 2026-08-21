# Case 01: Unreachable llama.cpp API

## Customer report

I started llama-server on `127.0.0.1:8080`, but requests from my Python program inside WSL fail with a connection error. The server appears to be running normally in its own terminal. I need to understand why the client cannot reach it and how to fix it.

## Observed behavior

```bash
curl http://127.0.0.1:8080/health
```

```text
curl: (7) Failed to connect to 127.0.0.1 port 8080: Could not connect to server
```

## Initial interpretation

According to the report the customer started the server on 127.0.0.1:8080 but when checking for the health it can't connect to that server, thus the error is solely proving that he is not able to reach it not that the server is actually running as they claimed.

## First hypothesis

The server might not actually be running.

## Hypothesis check

Later, curl.exe http://127.0.0.1:8080/health returned {"status":"ok"}.
That proved the server was running and reachable from the Windows side, even though Linux curl inside WSL could not reach it.

## Root cause

The reason why the customer wasn't able to reach the server was that plain curl was pointing toward WSL’s 127.0.0.1 while the server was running on Windows.

## Resolution

Change the client URL from http://127.0.0.1:8080 to the Windows host address reachable from WSL, in this case http://172.23.0.1:8080.

## Verification

```bash
curl http://172.23.0.1:8080/health
```

```json
{"status":"ok"}
```

## Prevention

To prevent this the server URL must stay configurable through `LLAMA_SERVER_URL` instead of hardcoding a specific WSL-address.

## Customer response

We traced the connection failure to the client using WSL’s loopback address (`127.0.0.1`) while `llama-server` was running on Windows. Configure the benchmark client to use the Windows host address reachable from WSL:

```bash
export LLAMA_SERVER_URL=http://172.23.0.1:8080/v1/chat/completions
```

After applying the change, the health endpoint returned `{"status":"ok"}`, confirming that WSL could reach the server.