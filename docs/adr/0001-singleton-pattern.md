# ADR-0001: Singleton Pattern for Global State

## Status

Accepted

## Context

HFL requires several global state objects that must be shared across the application:
- Model registry (loaded models metadata)
- Model pool (cached inference engines)
- Metrics collector
- Configuration

We need a pattern that provides:
- Single instance guarantee
- Thread-safe initialization
- Easy access from anywhere in the codebase
- Testability (ability to reset/mock)

## Decision

We use the Module-Level Singleton pattern with double-checked locking for thread safety.

```python
_instance: MyClass | None = None
_lock = threading.Lock()

def get_instance() -> MyClass:
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            _instance = MyClass()
        return _instance

def reset_instance() -> None:
    global _instance
    with _lock:
        _instance = None
```

## Consequences

### Positive

- Thread-safe initialization with minimal lock contention
- Fast path (no locking) for common case when instance exists
- Explicit `reset_*` functions for testing
- Clear module-level API

### Negative

- Global state makes pure functional programming harder
- Need to remember to reset in tests
- Cannot easily have multiple configurations simultaneously

### Neutral

- Pattern is well-known and understood
- Consistent across all singletons in the codebase

## Alternatives Considered

### Option A: Dependency Injection Container

Use a DI container to manage singleton lifecycle.

**Pros:**
- More testable
- Flexible configuration

**Cons:**
- Added complexity
- Learning curve for contributors
- Overkill for a CLI tool

### Option B: Class-Level Singleton (Metaclass)

Use a metaclass to enforce singleton behavior.

**Pros:**
- Automatic singleton enforcement

**Cons:**
- Less explicit
- Harder to reset for testing
- More "magic"

## References

- Python threading documentation
- Double-checked locking pattern
