# ADR-0003: GGUF as Default Model Format

## Status

Accepted

## Context

HFL needs to support running LLM models locally on consumer hardware. Models on HuggingFace Hub come in various formats:
- safetensors (PyTorch weights)
- GGUF (llama.cpp quantized format)
- Original PyTorch (.bin)
- ONNX

Consumer hardware has limited VRAM/RAM, making full-precision models impractical for most users. We need a default format that:
- Runs efficiently on consumer hardware
- Supports CPU and GPU inference
- Has good quality/size tradeoff
- Is widely supported

## Decision

We use GGUF (GGML Universal Format) as the default model format, with Q4_K_M quantization as the default quantization level.

**Conversion pipeline:**
```
HuggingFace (safetensors) -> FP16 GGUF -> Quantized GGUF (Q4_K_M)
```

**Quantization levels:**
| Level | Bits | Quality | Use Case |
|-------|------|---------|----------|
| Q2_K | ~2.5 | ~80% | Extreme compression |
| Q3_K_M | ~3.5 | ~87% | Low RAM |
| **Q4_K_M** | ~4.5 | ~92% | **Default - best balance** |
| Q5_K_M | ~5.0 | ~96% | High quality |
| Q6_K | ~6.5 | ~97% | Premium |
| Q8_0 | ~8.0 | ~98%+ | Maximum quality |
| F16 | 16.0 | 100% | No quantization |

## Consequences

### Positive

- Runs on consumer hardware (8GB RAM can run 7B models)
- CPU inference works well
- Metal/CUDA acceleration supported
- ~4x smaller than original weights
- Single file deployment

### Negative

- Conversion step required for most models
- Slight quality loss vs FP16
- llama.cpp required as dependency
- Not all architectures supported

### Neutral

- Users can choose different quantization
- Original weights preserved if needed
- Conversion is one-time operation

## Alternatives Considered

### Option A: Keep Original safetensors

Run models in original format using transformers.

**Pros:**
- No conversion needed
- Full precision

**Cons:**
- Requires 2-4x more memory
- Slower inference
- GPU required for practical use

### Option B: ONNX Format

Convert to ONNX for runtime optimization.

**Pros:**
- Good runtime performance
- Cross-platform

**Cons:**
- Complex conversion process
- Not all ops supported
- Quantization less mature

### Option C: AWQ/GPTQ Quantization

Use transformers-native quantization.

**Pros:**
- Native transformers support
- Good quality

**Cons:**
- Still requires GPU
- Higher memory than GGUF
- Less flexible

## References

- llama.cpp quantization documentation
- GGUF format specification
- Quantization quality benchmarks
