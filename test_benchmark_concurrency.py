from benchmarks import benchmark_concurrency


def test_run_streaming_request_measures_ttft(monkeypatch):
    stream_lines = [
        'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}',
        "",
        'data: {"choices":[{"delta":{"content":"GPU"}}]}',
        'data: {"choices":[{"delta":{"content":" inference"}}]}',
        "data: [DONE]",
    ]

    class FakeResponse:
        def raise_for_status(self):
            pass

        def iter_lines(self, decode_unicode=True):
            return iter(stream_lines)

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        benchmark_concurrency.requests,
        "post",
        fake_post,
    )

    times = iter([10.0, 10.25, 10.75])

    monkeypatch.setattr(
        benchmark_concurrency.time,
        "perf_counter",
        lambda: next(times),
    )

    result = benchmark_concurrency.run_streaming_request()

    assert result == {
        "time_to_first_token_seconds": 0.25,
        "total_request_latency_seconds": 0.75,
        "streaming_duration_seconds": 0.5,
    }
