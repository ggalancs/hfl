#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Local regression smoke — brings hfl up and RUNS every CLI command for real
# (real Hub network + real llama.cpp inference), not just ``--help``.
#
# This is the guard the ``quote_from_bytes`` search-download crash slipped
# past: unit tests mock the Hub and never exercise the real runtime path.
# NOT a CI test — it needs network, a downloadable model, and the compiled
# backends. Run it locally before cutting a release.
#
# Safety: it pulls a throwaway model whose registry name does NOT collide
# with yours, cleans up only the entries it created (via a before/after
# name diff), and never touches your models.
#
# Usage:  bash scripts/local-smoke.sh
#         HFL=~/.local/bin/hfl PORT=11599 bash scripts/local-smoke.sh
# Exit: 0 if every command ran without crashing, 1 otherwise.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

HFL="${HFL:-.venv/bin/hfl}"
PY="${PY:-.venv/bin/python}"
PORT="${PORT:-11599}"
export HFL_HOST="127.0.0.1"; export HFL_PORT="$PORT"

# Distinct name (won't collide with a user's SmolLM2 etc.), tiny, CPU-fast.
THROW_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
THROW_NAME=""; COPY_NAME=""

PASS=0; FAIL=0; SKIP=0; declare -a ROWS
pass(){ PASS=$((PASS+1)); ROWS+=("  ✅ $1"); }
skip(){ SKIP=$((SKIP+1)); ROWS+=("  ⏭️  $1  ($2)"); }
fail(){ FAIL=$((FAIL+1)); ROWS+=("  ❌ $1  ::  $2"); }

names(){ "$PY" -c "import json,os; print('\n'.join(m['name'] for m in json.load(open(os.path.expanduser('~/.hfl/models.json')))))" 2>/dev/null; }
crashed(){ echo "$1" | grep -qiE "traceback|quote_from_bytes|unhandledexception|AttributeError|TypeError|KeyError"; }

# exit-0 required
t(){ local l="$1"; shift; local o; o="$("$@" 2>&1)"; local rc=$?
  if [ $rc -eq 0 ]; then pass "$l"
  else fail "$l" "$(echo "$o"|grep -iE 'error|traceback|no such|not found'|head -1|cut -c1-110)"; fi; }
# "ran without crashing" + output marker (for cmds with domain-legit non-zero)
tran(){ local l="$1" marker="$2"; shift 2; local o; o="$("$@" 2>&1)"
  if crashed "$o"; then fail "$l" "CRASH: $(echo "$o"|grep -iE 'traceback|error'|head -1|cut -c1-100)"
  elif echo "$o"|grep -qiE "$marker"; then pass "$l"
  else fail "$l" "no esperado: $(echo "$o"|tail -1|cut -c1-100)"; fi; }

SERVE_PID=""
cleanup(){
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null
  [ -n "$COPY_NAME" ]  && printf 'y\n' | "$HFL" rm "$COPY_NAME"  >/dev/null 2>&1
  [ -n "$THROW_NAME" ] && printf 'y\n' | "$HFL" rm "$THROW_NAME" >/dev/null 2>&1
}
trap cleanup EXIT

echo "== hfl local smoke =="; "$HFL" --version 2>&1|head -1; echo "port: $PORT ; throwaway: $THROW_REPO"; echo

# 1. Read-only / diagnostics
for c in version help config check debug doctor list compliance-report compliance-dashboard; do t "$c" "$HFL" "$c"; done

# 2. Hub-backed
t    "search (resolve real)"  "$PY" -c "from hfl.hub.resolver import resolve; resolve('$THROW_REPO')"
t    "discover"               "$HFL" discover --limit 3
t    "recommend"              "$HFL" recommend --top 3
tran "draft-recommend"        "draft|candidate|recommend" "$HFL" draft-recommend "$THROW_REPO"

# 3. Lifecycle with a fresh, non-colliding throwaway (before/after diff)
BEFORE="$(names)"
t "pull (download)" "$HFL" pull "$THROW_REPO" -q Q4_K_M --skip-license
THROW_NAME="$(comm -13 <(echo "$BEFORE"|sort) <(names|sort) | head -1)"
echo "  (throwaway creado como: ${THROW_NAME:-NINGUNO})"
if [ -n "$THROW_NAME" ]; then
  t    "inspect <model>" "$HFL" inspect "$THROW_NAME"
  t    "show <model>"    "$HFL" show "$THROW_NAME"
  tran "verify <model>"  "VERIFY|round_trip|generation" "$HFL" verify "$THROW_NAME"
  t    "alias <model>"   "$HFL" alias "$THROW_NAME" hfl-smoke-alias
  COPY_NAME="${THROW_NAME}-smoketest"
  t    "cp <model>"      "$HFL" cp "$THROW_NAME" "$COPY_NAME"
  o="$(printf 'y\n' | "$HFL" rm "$COPY_NAME" 2>&1)"; if echo "$o"|grep -qi "deleted"; then pass "rm <copy>"; COPY_NAME=""; else fail "rm <copy>" "$(echo "$o"|tail -1|cut -c1-90)"; fi
else
  for c in inspect show verify alias cp rm; do fail "$c <model>" "throwaway no creado (¿colisión/pull falló?)"; done
fi

# 4. serve / inference / ps / stop
"$HFL" serve --port "$PORT" > /tmp/hfl-smoke-serve.log 2>&1 & SERVE_PID=$!
up=0; for i in $(seq 1 40); do curl -s "localhost:$PORT/healthz" >/dev/null 2>&1 && { up=1; break; }; sleep 1; done
if [ "$up" = 1 ]; then
  pass "serve (up on :$PORT)"
  if [ -n "$THROW_NAME" ]; then
    R="$(curl -s --max-time 90 "localhost:$PORT/api/chat" -H 'Content-Type: application/json' -d "{\"model\":\"$THROW_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word.\"}],\"stream\":false}")"
    echo "$R"|"$PY" -c "import sys,json;d=json.load(sys.stdin);sys.exit(0 if d.get('message',{}).get('content') else 1)" 2>/dev/null && pass "run/inference (/api/chat)" || fail "run/inference" "vacío: $(echo "$R"|cut -c1-90)"
    t "ps"          "$HFL" ps
    t "stop <model>" "$HFL" stop "$THROW_NAME"
  fi
else fail "serve" "no arrancó"; tail -3 /tmp/hfl-smoke-serve.log; fi

# 5. Subcommand leaves + guarded
t "lora list"      "$HFL" lora list
t "snapshot list"  "$HFL" snapshot list
t "mcp --help"     "$HFL" mcp --help
t "create (--help)"     "$HFL" create --help
t "pull-smart (--help)" "$HFL" pull-smart --help
t "login (--help)"      "$HFL" login --help
t "logout (--help)"     "$HFL" logout --help
skip "pull-smart (live)"  "descarga una variante completa; correr a mano"
skip "login/logout (live)" "tocaría tu token de HF"

echo; echo "================ RESULTADO ================"; printf '%s\n' "${ROWS[@]}"
echo "-------------------------------------------"; echo "  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
[ "$FAIL" -eq 0 ] && { echo "  ✅ todos los comandos ejecutan sin errores"; exit 0; } || { echo "  ❌ revisar fallos"; exit 1; }
