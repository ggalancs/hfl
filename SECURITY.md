# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in hfl, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. **Email** the maintainer at: [security contact to be added]
3. **Include** a detailed description of the vulnerability
4. **Include** steps to reproduce the issue
5. **Allow** reasonable time for a fix before public disclosure

## Security Considerations

### Token Handling

hfl is designed to handle HuggingFace tokens securely:

- Tokens are read **only** from environment variables (`HF_TOKEN`) or secure prompts
- Tokens are **never** persisted to disk, configuration files, or logs
- Tokens are held in memory only for the duration of the process

### Network Security

- All HuggingFace Hub connections use HTTPS
- The API server binds to `127.0.0.1` by default (localhost only)
- Exposing the server to `0.0.0.0` requires explicit confirmation

### Model License Compliance

hfl includes license verification to protect users from inadvertent license violations:

- Model licenses are checked and displayed before download
- License restrictions are stored with model metadata
- Users must explicitly accept non-permissive licenses

### AI Output Disclaimers

All AI-generated content includes disclaimers to inform users that:

- The content is AI-generated
- The content may be inaccurate or inappropriate
- Users are responsible for evaluating and using outputs

## Security Best Practices for Users

1. **Keep hfl updated** to receive security fixes
2. **Use environment variables** for tokens, not command-line arguments
3. **Do not expose** the API server to untrusted networks
4. **Review model licenses** before commercial use
5. **Validate AI outputs** before critical use
