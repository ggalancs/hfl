# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for hfl
#
# Build commands:
#   pyinstaller hfl.spec
#   HFL_PYI_ICON=packaging/macos/hfl.icns pyinstaller hfl.spec   # with an icon
#
# Every packaging (the release executables, the DMG, the MSI) builds from
# this spec, and each checks the result runs a model (platform_check.py
# --hfl dist/hfl --expect-backend llama.cpp) before shipping it: the 0.22.0
# executables and DMG could not run any model.
#
# Output: dist/hfl (or dist/hfl.exe on Windows)

import importlib.util
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

# Collect all rich submodules including unicode data
rich_imports = collect_submodules('rich')
rich_data = collect_data_files('rich')

# Collect hfl's own package data — the i18n locale JSONs
# (hfl/i18n/locales/*.json). The CLI loads them at startup via
# ``Path(__file__).parent / "locales"``; without bundling them the frozen
# binary crashes on every command with "Translation file not found".
hfl_data = collect_data_files('hfl')

# Packages whose native libraries are loaded at run time (ctypes / a Rust
# extension), which PyInstaller's import analysis does not see. Without
# them the executable could not run a model: ``import llama_cpp`` failed
# with FileNotFoundError (libllama missing), and downloads fell back from
# Xet to plain HTTP. Collected when the build environment has them.
native_datas, native_binaries, native_imports = [], [], []
for package in ('llama_cpp', 'hf_xet'):
    if importlib.util.find_spec(package) is not None:
        d, b, h = collect_all(package)
        native_datas += d
        native_binaries += b
        native_imports += h

# Detect platform
is_windows = sys.platform == 'win32'
is_macos = sys.platform == 'darwin'
is_linux = sys.platform.startswith('linux')

# Executable name
exe_name = 'hfl.exe' if is_windows else 'hfl'

# Hidden imports that PyInstaller doesn't detect automatically
hidden_imports = [
    # Core
    'hfl',
    'hfl.cli',
    'hfl.cli.main',
    'hfl.api',
    'hfl.api.server',
    'hfl.api.routes_native',
    'hfl.api.routes_openai',
    'hfl.api.middleware',
    'hfl.config',
    'hfl.exceptions',
    'hfl.engine',
    'hfl.engine.base',
    'hfl.engine.selector',
    'hfl.engine.llama_cpp',
    'hfl.engine.llama_server',
    'hfl.engine._child_guard',  # llama-server runs under it (hfl.utils.self_exec)
    'hfl.utils.self_exec',
    'hfl.converter',
    'hfl.converter.formats',
    'hfl.converter.gguf_converter',
    'hfl.hub',
    'hfl.hub.resolver',
    'hfl.hub.downloader',
    'hfl.hub.auth',
    'hfl.hub.license_checker',
    'hfl.models',
    'hfl.models.registry',
    'hfl.models.manifest',
    'hfl.models.provenance',
    # Dependencies
    'typer',
    'typer.main',
    'click',
    # Rich - collected dynamically via collect_submodules
    'pydantic',
    'pydantic.main',
    'fastapi',
    'uvicorn',
    'uvicorn.main',
    'uvicorn.config',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'starlette',
    'httpx',
    'sse_starlette',
    'huggingface_hub',
    'yaml',
    'json',
    # Encodings
    'encodings',
    'encodings.utf_8',
    'encodings.ascii',
    'encodings.latin_1',
]

# Additional data to include
datas = rich_data + hfl_data + native_datas

# Exclude heavy optional modules not needed for basic CLI
excludes = [
    'torch',
    'transformers',
    'vllm',
    'tensorflow',
    'keras',
    'numpy.distutils',
    'matplotlib',
    'scipy',
    'pandas',
    'PIL',
    'cv2',
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'sphinx',
    'setuptools',
    'wheel',
    'pip',
]

a = Analysis(
    ['src/hfl/cli/main.py'],
    pathex=[],
    binaries=native_binaries,
    datas=datas,
    hiddenimports=hidden_imports + rich_imports + native_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='hfl',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress with UPX if available
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # CLI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # The DMG and MSI builds pass their icon (.icns / .ico) through the
    # environment, so all three packagings build from this one spec.
    icon=os.environ.get('HFL_PYI_ICON') or None,
)
