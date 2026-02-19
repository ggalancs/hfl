# Contributing to hfl

Thank you for your interest in contributing to hfl! This document provides guidelines for contributions.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ggalancs/hfl
cd hfl

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests** for any new functionality
3. **Ensure all tests pass**: `pytest`
4. **Ensure code passes linting**: `ruff check src/ tests/`
5. **Update documentation** if needed
6. **Submit PR** with a clear description of changes

## Coding Standards

- Follow PEP 8 (enforced by ruff)
- Maximum line length: 100 characters
- Use type hints where practical
- Write docstrings for public APIs

## Compliance Module Requirements

Per the HRUL license, any contributions that modify the Compliance Modules must maintain their protective functionality:

1. **License Verification** - Must still verify and display model licenses
2. **Provenance Tracking** - Must still record conversion history
3. **AI Disclaimer** - Must still attach disclaimers to AI outputs
4. **Privacy Protection** - Must NOT persist tokens to disk
5. **Gating Respect** - Must NOT bypass HuggingFace gating

## Testing

- All new features require tests
- Maintain coverage above 80%
- Use mocks for external API calls (HuggingFace Hub)
- Use pytest fixtures from `conftest.py`

## Reporting Issues

- Use the GitHub issue tracker
- Include Python version, OS, and hfl version
- Provide minimal reproduction steps
- Include relevant error messages

## License

By contributing, you agree that your contributions will be licensed under the HRUL v1.0 license.
