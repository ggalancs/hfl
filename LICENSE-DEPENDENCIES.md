# Third-Party Licenses

hfl uses the following open-source dependencies. All dependencies use permissive licenses compatible with commercial use.

## Direct Dependencies

| Package | License | Purpose |
|---------|---------|---------|
| typer | MIT | CLI framework |
| rich | MIT | Terminal formatting and tables |
| huggingface-hub | Apache 2.0 | HuggingFace API client |
| fastapi | MIT | REST API framework |
| uvicorn | BSD-3-Clause | ASGI server |
| pydantic | MIT | Data validation |
| httpx | BSD-3-Clause | HTTP client |
| sse-starlette | BSD-3-Clause | Server-Sent Events |
| pyyaml | MIT | YAML configuration |

## Optional Dependencies

| Package | License | Purpose |
|---------|---------|---------|
| llama-cpp-python | MIT | llama.cpp bindings for GGUF inference |
| transformers | Apache 2.0 | HuggingFace Transformers library |
| torch | BSD-3-Clause | PyTorch (for Transformers backend) |
| accelerate | Apache 2.0 | Model acceleration utilities |
| sentencepiece | Apache 2.0 | Tokenization |
| vllm | Apache 2.0 | High-performance inference |
| gguf | MIT | GGUF file format tools |

## Tools Used for Conversion

| Tool | License | Purpose |
|------|---------|---------|
| llama.cpp | MIT | Model conversion and quantization |
| bitsandbytes | MIT | Dynamic GPU quantization |

## License Compatibility

All dependencies use permissive licenses:

- **MIT**: Free use, modification, distribution with attribution
- **Apache 2.0**: Same as MIT, plus patent grant
- **BSD-3-Clause**: Same as MIT, with non-endorsement clause

No dependencies use copyleft licenses (GPL, AGPL, LGPL).

## Verification

To verify dependency licenses in your installation:

```bash
pip install pip-licenses
pip-licenses --format=table --with-urls
```

## License Check CI

This project includes automated license checking in CI to prevent accidental introduction of copyleft dependencies. See `.github/workflows/license-check.yml`.

---

*Last updated: February 2026*
