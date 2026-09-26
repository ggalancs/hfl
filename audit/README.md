# HFL local audit

An end-to-end audit of HFL on your own machine: every CLI command, every API
route, every install extra, every environment variable, every engine and every
way of installing HFL — each one **run for real**. It installs HFL the way a
user would, downloads small models, starts real servers and talks to them.

Nothing is mocked. A check passes only on evidence: an answer, a status code,
a file, a log line, and several carry a control (the same thing measured
without the setting) so a check that cannot fail does not pass.

## What you need

- Python ≥ 3.10 with `httpx` to run the harness (the repo's dev venv has it:
  `pip install -e ".[dev]"`).
- [`uv`](https://docs.astral.sh/uv/) — `--setup` and section C build the
  wheel and venvs with it.
- ~20 GB of disk and a network connection (models come from the Hugging Face
  Hub; all are Apache-2.0 or MIT, so no license is accepted on your behalf).
- Optional, each unlocking checks that otherwise report `NO COMPROBABLE AQUÍ`:
  `llama-server` on `PATH` (`brew install llama.cpp`), Docker, Homebrew,
  `cmake` (safetensors → GGUF conversion), a speech synthesiser (macOS `say`,
  or `espeak-ng`), Apple Silicon (MLX), Linux + NVIDIA (vLLM).

## Run it

```bash
# 1. Build this checkout's wheel and install it into <work>/venv (all extras
#    that exist on this platform) and <work>/venv-core (no extras).
python audit/local_audit.py --work ~/hfl-audit --setup

# 2. Every check (a few hours; see "Long runs" below).
python audit/local_audit.py --work ~/hfl-audit

# A section, or some checks — a check's prerequisites run first.
python audit/local_audit.py --work ~/hfl-audit --only D
python audit/local_audit.py --work ~/hfl-audit --only B3,D14,E8
python audit/local_audit.py --work ~/hfl-audit --list
```

Results accumulate in `<work>/results.json` (a re-run of some checks updates
only those) and every run writes `<work>/REPORT.md`: the counts, what is not
OK first, then every check with its evidence.

| status | means |
|---|---|
| `OK` | checked and correct; the evidence says how |
| `ROTO` | broken: the evidence is what came back instead |
| `NO COMPROBABLE AQUÍ` | cannot be checked on this machine, and why (never counted as OK) |
| `REQUIERE PERMISO` | would publish or install something on the owner's behalf; not done |

The exit code is 1 when anything is `ROTO`.

## Sections

| | what | checks |
|---|---|---|
| A | CLI commands | `audit_checks/a_cli.py` |
| B | API routes (OpenAI, Anthropic, Ollama, native) | `audit_checks/b_api.py`, speech/image routes in `e_engines.py` |
| C | each install extra, alone, in a fresh venv | `audit_checks/c_extras.py` |
| D | each environment variable, and whether it is documented | `audit_checks/d_env.py` |
| E | each engine through the same chat checks over the three APIs | `audit_checks/e_engines.py` |
| F | ways to install and run: wheel, extras, Docker, Homebrew, tray, PyInstaller | `audit_checks/f_install.py` |

The chat web UI (`/ui`) is checked by hand in a browser: open
`http://127.0.0.1:11434/ui` on a server with a few models and hold a
conversation, reload, start a new one.

## Isolation

The audit never touches your own HFL or Hugging Face setup: it runs with its
own `HFL_HOME` and `HF_HOME` under `<work>`, strips every `HFL_*`, `OLLAMA_*`
and `HF_TOKEN` variable from the environment, and stops every server it starts
when its check ends. Nothing is pushed to the Hub. Delete `<work>` when done.

## Long runs

A full run takes hours. Keep the machine awake (macOS: `caffeinate -dimsu`;
it does not survive closing the lid) and give it an upper bound that does not
depend on the audit itself:

```bash
caffeinate -dimsu python audit/watchdog.py 14400 ~/hfl-audit/run.log -- \
    python -u audit/local_audit.py --work ~/hfl-audit
```

`watchdog.py` runs the audit in its own process group, writes its output
unbuffered to the log, and kills the whole group — servers included — when
the time is up.

## Writing a check

```python
from local_audit import Audit, Parts, check, expect


@check("B99", "POST /api/something", needs=("B3",))  # needs: run B3 first
def b99(a: Audit) -> str:
    with a.server() as base:  # a real `hfl serve`
        out = httpx.post(base + "/api/something", json={...})
    expect(out.status_code == 200, out.text[:200])  # ROTO with this evidence
    return "what was seen"  # the OK evidence
```

`Parts` checks several things and reports every failure, not just the first;
raise `Uncheckable(why)` when the machine cannot run it, and
`need_llama_server()` / `need_apple_silicon(what)` for the common cases. Make
the check able to fail: run it once against something broken and watch it
turn `ROTO`.
