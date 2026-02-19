# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for hfl
#
# Build commands:
#   pyinstaller hfl.spec
#
# Output: dist/hfl (or dist/hfl.exe on Windows)

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Collect all rich submodules including unicode data
rich_imports = collect_submodules('rich')
rich_data = collect_data_files('rich')

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
datas = rich_data

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
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports + rich_imports,
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
    icon=None,  # Add icon if desired: icon='assets/hfl.ico'
)
