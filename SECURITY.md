# Security Policy

## Supported Versions

| Version                    | Supported          |
| -------------------------- | ------------------ |
| Latest release (0.27.x)    | :white_check_mark: |
| Older releases             | :x:                |

Fixes go into the next release; please upgrade to the latest version.

## Reporting a Vulnerability

If you discover a security vulnerability in hfl, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. **Report it privately** on GitHub: [Report a vulnerability](https://github.com/ggalancs/hfl/security/advisories/new)
   (the repository's **Security** tab → **Report a vulnerability**). Only the maintainer sees it.
3. **Include** a detailed description of the vulnerability
4. **Include** steps to reproduce the issue
5. **Allow** reasonable time for a fix before public disclosure

## Security Considerations

### Token Handling

hfl is designed to handle HuggingFace tokens securely:

- Tokens are read from the `HF_TOKEN` environment variable, or from the token `hfl login` saved
- `hfl login` stores the token where the Hugging Face libraries keep it (`$HF_HOME/token`, by default
  `~/.cache/huggingface/token`), as `hf auth login` does; `hfl logout` removes it. Without
  `hfl login`, nothing is written to disk
- HFL keeps no copy of the token in its own configuration, and masks it in the errors it logs
  from Hub uploads (`hfl push`)

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
