# Packaging

Files for the package managers HFL is not in yet. Publishing them is the
maintainer's step: each one goes out under the maintainer's name.

## Homebrew — `homebrew/hfl.rb`

A tap formula: HFL and its Python dependencies in a virtualenv, GGUF models
served by Homebrew's `llama.cpp` (without llama-cpp-python, HFL falls back to
`llama-server`; that fallback ships after 0.21.0, so the formula serves GGUF
from the next release on).

Verified locally on 2026-09-25 (macOS 26, Apple Silicon): `brew install
--build-from-source`, `brew test` and `brew audit --strict --online` pass from
a local tap; a Homebrew-Python environment with the current code served a GGUF
through Homebrew's `llama-server`.

To publish:

1. Create the public repository `ggalancs/homebrew-hfl` with `Formula/hfl.rb`
   copied from here.
2. Users then run `brew install ggalancs/hfl/hfl`.
3. For a new release, run the **Homebrew tap** workflow by hand: it rewrites the
   formula's url and sha256 from PyPI and opens a PR on the tap (it needs the
   `HOMEBREW_TAP_TOKEN` secret). If the dependencies changed, regenerate the
   resources first with `brew update-python-resources` — at least a day after
   the PyPI upload: Homebrew ignores packages younger than that.

## winget — `winget/manifests/g/ggalancs/HFL/0.21.0/`

Manifests for the 0.21.0 MSI of the GitHub release (sha256 computed from the
published file). Validated against winget's 1.9.0 JSON schemas; not run
through `winget validate` / `winget install`, which need Windows.

To publish: on Windows, `winget validate` and `winget install --manifest` the
folder, then open a pull request adding it to
[microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs) under the same
path. The identifier `ggalancs.HFL` can still be changed before that first PR.
