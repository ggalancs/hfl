# Packaging

Files for the package managers HFL is not in yet. Publishing them is the
maintainer's step: each one goes out under the maintainer's name.

## Homebrew — `homebrew/hfl.rb`

A tap formula: HFL and its Python dependencies in a virtualenv, GGUF models
served by Homebrew's `llama.cpp` (without llama-cpp-python, HFL serves GGUF
through `llama-server`, since 0.22.0). The formula points at the 0.28.0 sdist
on PyPI (sha256 checked against a download and PyPI's digest). 0.26.0
declared `psutil`, which earlier releases imported without declaring; its
resource was added then (the sdist builds with Homebrew's Python 3.14). The
other resources are unchanged from 0.21.0 and within 0.28.0's ranges (0.28.0
only raised ceilings: huggingface-hub `<3.0`, fastapi `<0.163`, pydantic
`<2.15`; the formula keeps 1.33.0, 0.141.1 and 2.13.5).

Verified locally on 2026-10-15 with Homebrew 7.0.9 (macOS 26, Apple Silicon),
the formula at 0.27.1 from PyPI: `brew install --build-from-source`,
`brew test` and `brew audit --strict --online` pass from a local tap, and the
installed `hfl` pulled a GGUF and served it through Homebrew's `llama-server`
(an answer to a chat request). Homebrew 6.0.18 could not install it: it
stopped while upgrading the dependencies ("already locked … openssl@3").
The 0.28.0 formula has not been installed with `brew` yet.

To publish:

1. Create the public repository `ggalancs/homebrew-hfl` with `Formula/hfl.rb`
   copied from here.
2. Users then run `brew install ggalancs/hfl/hfl`.
3. For a new release, run the **Homebrew tap** workflow by hand: it rewrites the
   formula's url and sha256 from PyPI and opens a PR on the tap (it needs the
   `HOMEBREW_TAP_TOKEN` secret). If the dependencies changed, regenerate the
   resources first with `brew update-python-resources` — at least a day after
   the PyPI upload: Homebrew ignores packages younger than that.

## winget — `winget/manifests/g/ggalancs/HFL/0.28.0/`

Manifests for the 0.28.0 MSI of the GitHub release, schema 1.12.0 (what
winget-pkgs asks for now); its sha256 equals a download's and GitHub's
digest. The 0.27.1 manifests were checked on a real Windows 10 on
2026-10-15: `winget validate` passed; installed silently with msiexec over
HFL 0.23.0, the MSI upgraded it (the old product found and removed), served a
GGUF through its bundled llama-server, and uninstalled cleanly. The 0.28.0
manifests have not been through `winget validate` yet (only their version,
URL and hash changed). `winget install --manifest` itself was not run: it
needs the `LocalManifestFiles` admin setting.

To publish: on Windows, `winget validate` and `winget install --manifest` the
folder, then open a pull request adding it to
[microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs) under the same
path. The identifier `ggalancs.HFL` can still be changed before that first PR.
