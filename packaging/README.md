# Packaging

Files for the package managers HFL is not in yet. Publishing them is the
maintainer's step: each one goes out under the maintainer's name.

## Homebrew — `homebrew/hfl.rb`

A tap formula: HFL and its Python dependencies in a virtualenv, GGUF models
served by Homebrew's `llama.cpp` (without llama-cpp-python, HFL serves GGUF
through `llama-server`, since 0.22.0). The formula points at the 0.27.0 sdist
on PyPI (sha256 checked against a download and PyPI's digest). 0.26.0
declared `psutil`, which earlier releases imported without declaring; its
resource was added then (the sdist builds with Homebrew's Python 3.14). The
other resources are unchanged from 0.21.0 and within 0.27.0's ranges (0.27.0
only raised fastapi's ceiling to `<0.143`; the formula's fastapi is 0.141.1).

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

## winget — `winget/manifests/g/ggalancs/HFL/0.27.0/`

Manifests for the 0.27.0 MSI of the GitHub release (sha256 computed from the
published file, and equal to the one GitHub reports for it). The same
manifests, at 0.25.0 and differing only in version, URL and hash, passed
`winget validate` on a real Windows 10 (local audit F7,
which also builds the MSI and extracts it without installing); `winget
install` from winget-pkgs needs the PR there. The MSI's executable ran a
model on GitHub's Windows runner before it was published.

To publish: on Windows, `winget validate` and `winget install --manifest` the
folder, then open a pull request adding it to
[microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs) under the same
path. The identifier `ggalancs.HFL` can still be changed before that first PR.
