# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
#
# Homebrew formula for HFL — "Run HuggingFace models locally, Ollama-compatible".
#
# Installed via the tap:
#
#     brew tap ggalancs/hfl
#     brew install hfl
#
# The formula pulls the wheel from PyPI (or a pinned tarball mirrored
# on GitHub Releases) so users get whichever llama-cpp-python wheel
# matches their Python and hardware. The shebang runs ``python3`` from
# Homebrew's own site-packages.
#
# Template version/URLs are patched in place by
# ``.github/workflows/homebrew.yml`` when a new ``v*`` tag lands. The
# placeholder values below keep the formula lintable today (``brew
# audit --strict hfl`` passes).

class Hfl < Formula
  include Language::Python::Virtualenv

  desc "Run HuggingFace models locally, Ollama-compatible"
  homepage "https://github.com/ggalancs/hfl"
  url "https://files.pythonhosted.org/packages/source/h/hfl/hfl-0.10.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "HRUL-1.0"
  head "https://github.com/ggalancs/hfl.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "rust" => :build
  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  service do
    run [opt_bin/"hfl", "serve", "--host", "127.0.0.1", "--port", "11434"]
    keep_alive successful_exit: false
    log_path var/"log/hfl.log"
    error_log_path var/"log/hfl.err.log"
  end

  test do
    assert_match "hfl v", shell_output("#{bin}/hfl version")
  end
end
