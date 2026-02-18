#!/bin/bash
# Build hfl executable for the current platform
#
# Usage:
#   ./scripts/build_executable.sh [--with-llama]
#
# Output: dist/hfl (or dist/hfl.exe on Windows)

set -e

WITH_LLAMA=false
if [[ "$1" == "--with-llama" ]]; then
    WITH_LLAMA=true
    echo "Building WITH llama-cpp-python support"
fi

echo "=========================================="
echo "   hfl - Build Executable"
echo "=========================================="
echo ""

# Detect platform
case "$(uname -s)" in
    Linux*)     PLATFORM="linux";;
    Darwin*)    PLATFORM="macos";;
    CYGWIN*|MINGW*|MSYS*) PLATFORM="windows";;
    *)          PLATFORM="unknown";;
esac

echo "Platform: $PLATFORM"
echo ""

# Install dependencies
echo "[1/4] Installing build dependencies..."
pip install pyinstaller -q

echo "[2/4] Installing hfl..."
pip install -e . -q

if $WITH_LLAMA; then
    echo "[2.5/4] Installing llama-cpp-python..."
    if [[ "$PLATFORM" == "macos" ]]; then
        CMAKE_ARGS="-DGGML_METAL=ON" pip install llama-cpp-python
    else
        pip install llama-cpp-python
    fi
fi

# Clean previous builds
echo "[3/4] Cleaning previous builds..."
rm -rf build/ dist/

# Build
echo "[4/4] Building executable..."
pyinstaller hfl.spec --clean

echo ""
echo "=========================================="
echo "   Build Complete!"
echo "=========================================="
echo ""

if [[ "$PLATFORM" == "windows" ]]; then
    echo "Executable: dist/hfl.exe"
    echo ""
    echo "Test with: ./dist/hfl.exe version"
else
    echo "Executable: dist/hfl"
    echo ""
    echo "Test with: ./dist/hfl version"
    echo ""
    echo "Install system-wide:"
    echo "  sudo cp dist/hfl /usr/local/bin/"
fi
