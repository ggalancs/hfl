# ADR-0006: Rate Limiting Strategy

## Status

Accepted

## Context

HFL's API server needs protection against:
- Accidental DoS from misconfigured clients
- Resource exhaustion from too many concurrent requests
- Abuse when exposed to untrusted networks

Rate limiting requirements:
- Must be enabled by default for security
- Should not impact normal usage
- Configurable for different deployment scenarios
- Minimal overhead

## Decision

We implement sliding window rate limiting at the middleware level, enabled by default.

**Default configuration:**
```python
rate_limit_enabled: bool = True
rate_limit_requests: int = 60  # per window
rate_limit_window: int = 60    # seconds (1 minute)
```

**Algorithm: Sliding Window Log**
```python
class RateLimitMiddleware:
    def _is_rate_limited(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self._window_seconds

        # Clean old entries
        self._request_counts[client_ip] = [
            ts for ts in self._request_counts[client_ip]
            if ts > window_start
        ]

        # Check limit
        if len(self._request_counts[client_ip]) >= self._max_requests:
            return True

        self._request_counts[client_ip].append(now)
        return False
```

**Client identification:**
1. X-Forwarded-For header (if behind proxy)
2. X-Real-IP header (nginx)
3. Direct client IP (fallback)

**Response when limited:**
- Status: 429 Too Many Requests
- Header: Retry-After: N
- Body: {"error": "Rate limit exceeded"}

## Consequences

### Positive

- Protection against abuse by default
- Configurable per deployment
- Minimal memory footprint
- Per-client fairness

### Negative

- May affect legitimate high-throughput clients
- Memory grows with unique clients
- No distributed rate limiting (single instance)

### Neutral

- Health endpoints exempt from limiting
- Can be disabled if needed
- Tests reset state between runs

## Alternatives Considered

### Option A: Token Bucket

Classic token bucket algorithm.

**Pros:**
- Allows burst traffic
- Smooth rate over time

**Cons:**
- More complex state
- Burst may overwhelm server

### Option B: Fixed Window

Simple counter reset each window.

**Pros:**
- Very simple
- Low overhead

**Cons:**
- Edge case: 2x burst at window boundaries
- Less fair

### Option C: External Rate Limiter

Use Redis or external service.

**Pros:**
- Distributed limiting
- Shared state across instances

**Cons:**
- External dependency
- Deployment complexity
- Overkill for local tool

## References

- Rate limiting algorithms comparison
- FastAPI middleware documentation
- HTTP 429 specification
