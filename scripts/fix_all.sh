#!/bin/bash
# Script maestro para corregir todos los problemas detectados
# Uso: ./scripts/fix_all.sh [--dry-run]

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== MODO DRY-RUN: No se aplicarán cambios ==="
fi

echo "=========================================="
echo "   hfl - Script de Corrección Automática"
echo "=========================================="
echo ""

# 1. Fix imports
echo "[1/5] Corrigiendo ordenamiento de imports..."
if $DRY_RUN; then
    ruff check src/ --select=I --diff
else
    ruff check src/ --select=I --fix
fi

# 2. Format code
echo ""
echo "[2/5] Formateando código..."
if $DRY_RUN; then
    ruff format src/ --diff
else
    ruff format src/
fi

# 3. Run all linting
echo ""
echo "[3/5] Verificando linting completo..."
ruff check src/ || true

# 4. Type checking
echo ""
echo "[4/5] Verificando tipos..."
mypy src/hfl --ignore-missing-imports || true

# 5. Run tests
echo ""
echo "[5/5] Ejecutando tests..."
pytest --cov=hfl --cov-report=term-missing --cov-fail-under=80

echo ""
echo "=========================================="
echo "   Correcciones completadas"
echo "=========================================="
