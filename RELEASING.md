# Releasing HFL

How a version goes out. Every workflow is `workflow_dispatch` only, run by
hand from the Actions tab or with `gh workflow run`; nothing publishes on a
push or a tag, and nothing should (Actions minutes are paid, and a release
is a decision). Each rule below comes from a release that went wrong
without it.

## 1. Before tagging: everything green locally

A workflow is launched on GitHub only once what it does has passed on a
local machine.

1. **The gate**: `bash scripts/ci-local.sh` (lint, format, mypy, the test
   suite, in a venv without `[llama]`, as CI has it). Run it unpiped:
   piping it through `tail` hides its exit code.
2. **What `ci-local.sh` does not cover**, which `lint.yml` checks:
   `ruff check src/` and `mypy src/hfl` — in **both** the dev venv (with the
   optional backends) and `.venv-ci` (without): they fail in different ways.
   0.16.1 shipped a red type gate checked in one of them only.
3. **The audit on three platforms**, 0 broken:
   `python audit/local_audit.py --work ~/hfl-audit-X --setup`, then the same
   without `--setup` (see `audit/README.md`): macOS, Linux (the Docker
   recipe in `audit/README.md`) and Windows. "Not checkable here" is fine;
   "broken" is not. Do not bump the version while an audit runs (B36
   compares the built wheel with `pyproject.toml`).
4. **Every changed workflow, reproduced locally.** What the packaging
   workflows check is in scripts that run anywhere:
   - `scripts/image_check.py <image>` — the Docker image as a user runs it;
     build both variants (`HFL_EXTRAS=llama` and `all`) for both
     architectures (`docker build --platform linux/amd64` is emulated on an
     Apple Silicon Mac, and enough).
   - `scripts/fetch_llama_server.py build/llama.cpp`, then
     `HFL_PYI_LLAMA_CPP=build/llama.cpp pyinstaller hfl.spec`, then
     `scripts/platform_check.py --hfl dist/hfl --expect-backend llama-server`
     and again with `HFL_LLM_LIBRARY=llama-cpp` and
     `--expect-backend llama.cpp` — on each platform the executables ship
     for, with no `llama-server` of your own on the PATH (it would hide a
     missing bundled one). macOS Intel builds under Rosetta with an x86-64
     (or universal2) Python; Linux x86-64 in an `ubuntu:24.04` container.
   - Windows (MSI and executable) on a Windows machine: a macOS or Linux
     run cannot show a Windows-only break (a bundled DLL that shadowed
     another one broke every request on Windows only).
   - Then, and only then, run the changed workflows on `main` (no tag): they
     build and check without publishing a release (Docker pushes only `dev`
     and `sha-…` tags).

## 2. The release

1. Bump the version in **`pyproject.toml` and `src/hfl/__init__.py`** — the
   only two version files (`hfl.wxs` takes it at build time; `uv.lock` is
   not tracked). In the dev venv, `uv pip install -e . --no-deps` again, or
   `importlib.metadata` keeps the old number.
2. Move `CHANGELOG.md`'s `[Unreleased]` to `[X.Y.Z] - date`.
3. Commit `chore(release): X.Y.Z`, tag `vX.Y.Z`, push `main` and the tag.
4. Run the workflows against the **tag** (they take the version from it):
   1. `Build Executables` with `-f version=vX.Y.Z`, **first and alone**: it
      creates the GitHub Release (as a draft) that the DMG and MSI attach to.
   2. Then, together: `Publish to PyPI` (`publish-pypi.yml`), `Docker`,
      `macOS DMG`, `Windows MSI`; and `Pages` from `main`.
   - **Never `release.yml` as well as `publish-pypi.yml`**: both publish to
     PyPI, and the second fails.
   - `Homebrew tap` needs the tap repository, which does not exist yet.

## 3. After publishing: check every surface

- PyPI: `pip install hfl==X.Y.Z` in a clean venv; `hfl version`. (The JSON
  API can serve a stale cache; `/simple/hfl/` is what pip reads.)
- GHCR: pull `ghcr.io/ggalancs/hfl:X.Y.Z` and run
  `scripts/image_check.py` on it; `latest` must be the slim image and
  `latest-all` the full one.
- GitHub Release: marked Latest, with the four executables, the DMG, the
  MSI and the checksums.
- Pages answers.
- **Every version reference in the repository**: `git grep` the previous
  version and update what names the current one — the Homebrew formula
  (`packaging/homebrew/hfl.rb`: sdist URL and sha256), the winget manifests
  (`packaging/winget/`: MSI URL and sha256), `packaging/README.md`, docs
  that say which version was tested. Hashes from the published files,
  cross-checked with PyPI's and GitHub's digests. Keep history and measured
  data as they are.
- Publishing the Homebrew tap and the winget-pkgs pull request goes out
  under the owner's name: the files are updated here, the publication is
  the owner's step.
