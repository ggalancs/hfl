# Stability contract

What HFL promises not to change under you, and how it changes what it
does change.

## What is public and stable

- **The CLI**: every command (including `hfl install …` and `hfl sessions …`),
  its arguments and its `--options`, and their meaning. Exit code 0 means
  success; non-zero means it did not do what was asked.
- **The HTTP API**: every route the server lists in its OpenAPI schema
  (`/openapi.json`), by method and path — the OpenAI-compatible (`/v1/…`),
  Anthropic-compatible (`/v1/messages…`) and Ollama-compatible (`/api/…`)
  surfaces, and HFL's own routes (`/healthz`, `/metrics`, `/api/pull/smart`, …)
  — with the request fields they accept and the response fields they return.
  New optional request fields and new response fields can appear: a client
  must ignore fields it does not know.
- **The environment variables** documented in [env-vars.md](env-vars.md)
  (`HFL_*`, and the `OLLAMA_*` aliases listed there), with their defaults.
- **The model registry** (`~/.hfl/models.json`): the fields of each entry.
  A newer HFL reads an older registry, and an older one ignores the fields
  it does not know; new fields can appear.

`tests/stability/surface.json` records all of this as the code has it, and
`tests/test_stability_contract.py` fails when something recorded disappears
without a deprecation, or when something new appears without being recorded.

## What is not

- HFL's Python modules (`import hfl…`): internal, they change at any time.
  Plugins use the entry points described in [plugins.md](plugins.md).
- Human-readable output: the CLI's tables and messages, log lines, error
  message text (match on the HTTP status and the error `code`, not on the
  message), `/ui`.
- Which engine serves a model when none was chosen, and performance.
- Anything not listed above.

## How things change

- **Until 1.0** (now): a minor version (0.x → 0.y) may change the contract;
  the CHANGELOG says so under *Changed* or *Removed*, and a deprecation comes
  first whenever possible.
- **From 1.0**: [Semantic Versioning](https://semver.org). A patch release
  fixes; a minor release adds; only a major release removes or changes the
  meaning of anything in the contract.
- **Deprecation before removal**, announced in at least one minor release
  before the removal:
  - a route answers with the `Deprecation` header (RFC 8594), a `Link` to
    its replacement, and a `Sunset` date once one is set; it is marked
    deprecated in `/openapi.json`;
  - a command, an option or a variable logs a warning naming its
    replacement;
  - the CHANGELOG lists it under *Deprecated*, and
    `tests/stability/surface.json` records it under `deprecated` with the
    version and the replacement.

Deprecated now:

| What | Since | Use instead |
|---|---|---|
| `POST /api/embeddings` | 0.21.0 | `POST /api/embed` |

## For contributors

A change that adds a command, an option, a route or a variable runs
`python scripts/stability_surface.py --write` and commits the updated
`tests/stability/surface.json`: the addition joins the contract on purpose.
A removal deprecates first (above).
