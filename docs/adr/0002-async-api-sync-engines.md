# ADR-0002: Async API with Sync Inference Engines

## Status

Accepted

## Context

HFL provides a FastAPI-based REST API that needs to handle multiple concurrent requests. However, the underlying inference engines (llama-cpp-python, transformers) are synchronous and perform blocking operations during model loading and inference.

The challenge is bridging async HTTP handling with sync inference without:
- Blocking the event loop
- Losing concurrency benefits
- Adding excessive complexity

## Decision

We use `asyncio.to_thread()` to run synchronous engine operations in a thread pool, combined with an `AsyncEngineWrapper` class for streaming operations.

**Non-streaming operations:**
```python
async def generate(request):
    result = await asyncio.to_thread(engine.generate, prompt, config)
    return result
```

**Streaming operations:**
```python
async def generate_stream(prompt, config):
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def producer():
        for token in engine.generate_stream(prompt, config):
            loop.call_soon_threadsafe(queue.put_nowait, token)
        loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.create_task(asyncio.to_thread(producer))

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token
```

## Consequences

### Positive

- Event loop stays responsive
- Multiple requests can be processed concurrently
- Simple pattern, easy to understand
- Works with any sync engine without modification

### Negative

- Thread pool has limited size (default: CPU count)
- Context switching overhead
- Streaming requires queue coordination

### Neutral

- Memory usage slightly higher due to thread overhead
- Latency unchanged for actual inference

## Alternatives Considered

### Option A: Native Async Engines

Require engines to implement async interfaces natively.

**Pros:**
- No thread overhead
- True async all the way down

**Cons:**
- Major rewrite of engine code
- External libraries (llama-cpp) don't support async
- Significant maintenance burden

### Option B: Process Pool

Use multiprocessing instead of threading.

**Pros:**
- True parallelism (no GIL)
- Process isolation

**Cons:**
- Much higher overhead
- Complex IPC for streaming
- Memory duplication for models

### Option C: Single-Threaded (No Async)

Use a simple sync server (Flask/Gunicorn).

**Pros:**
- Simpler code
- No async complexity

**Cons:**
- No concurrency
- Can't handle multiple requests
- Poor for streaming

## References

- Python asyncio documentation
- FastAPI background tasks documentation
- llama-cpp-python threading model
