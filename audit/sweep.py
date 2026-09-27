# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Compatibility sweep: the Hub's most downloaded text-generation models,
each pulled and asked a question through a real ``hfl serve``.

HFL's claim is "any model on the Hub"; this measures it on the ones people
actually use. Two lists, most downloaded first:

- safetensors repos (what HFL runs through MLX, Transformers, or converts
  to GGUF, depending on the machine), up to ``--max-params`` billion;
- GGUF repos, pulled at Q4_K_M, up to ``--max-gguf-gb``.

Only models under Apache-2.0 or MIT that are not gated: nothing is accepted
on anyone's behalf and no token is used. Each model is removed after its
turn, so the disk holds one at a time. Results go to ``<work>/sweep.json``
(a re-run skips what is there) and ``<work>/SWEEP.md``.

    python audit/sweep.py --work ~/hfl-audit --n 15
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_audit import QUESTION, Audit, Broken  # noqa: E402

HUB = "https://huggingface.co/api/models"
LICENSES = {"license:apache-2.0", "license:mit"}


def _listing(extra: dict[str, Any]) -> list[dict]:
    params: list[tuple[str, str]] = [
        ("pipeline_tag", "text-generation"),
        ("sort", "downloads"),
        ("direction", "-1"),
        ("limit", "300"),
        *[("expand[]", f) for f in ("downloads", "gated", "tags", "safetensors")],
        *extra.items(),
    ]
    out = httpx.get(HUB, params=params, timeout=60)
    out.raise_for_status()
    return list(out.json())


def _open(model: dict) -> bool:
    return not model.get("gated") and bool(LICENSES & set(model.get("tags") or []))


def _named_size_b(repo: str) -> float | None:
    """The size a repo's name states ("…-35B"): its safetensors metadata can
    undercount (Ornith-1.0-35B reported 0.0 B, measured)."""
    found = re.findall(r"(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", repo.split("/")[-1])
    return max(float(x) for x in found) if found else None


def safetensors_candidates(n: int, max_params_b: float) -> list[dict]:
    picked = []
    for m in _listing({}):
        total = (m.get("safetensors") or {}).get("total")
        if not _open(m) or "gguf" in (m.get("tags") or []) or not total:
            continue
        named = _named_size_b(m["id"])
        if total > max_params_b * 1e9 or (named is not None and named > max_params_b):
            continue
        picked.append(
            {
                "repo": m["id"],
                "kind": "safetensors",
                "params_b": round(total / 1e9, 2),
                "downloads": m.get("downloads", 0),
            }
        )
        if len(picked) == n:
            break
    return picked


def gguf_candidates(n: int, max_gb: float) -> list[dict]:
    picked = []
    for m in _listing({"filter": "gguf"}):
        if not _open(m):
            continue
        tree = httpx.get(f"{HUB}/{m['id']}/tree/main", timeout=60)
        if tree.status_code != 200:
            continue
        files = [f for f in tree.json() if f.get("path", "").lower().endswith(".gguf")]
        q4 = [f for f in files if "q4_k_m" in f["path"].lower() and "mmproj" not in f["path"]]
        if not q4 or q4[0].get("size", 0) > max_gb * 1024**3:
            continue
        size_gb = round(q4[0]["size"] / 1024**3, 2)
        downloads = m.get("downloads", 0)
        picked.append({"repo": m["id"], "kind": "gguf", "size_gb": size_gb, "downloads": downloads})
        if len(picked) == n:
            break
    return picked


def _registered(a: Audit) -> list[str]:
    path = a.home / "models.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    models = data if isinstance(data, list) else data.get("models", [])
    return [m["name"] for m in models]


def try_model(a: Audit, base: str, item: dict) -> dict:
    """Pull, ask, record, remove. Every step's outcome is kept: a model that
    cannot be pulled is a different finding from one that answers wrong."""
    result: dict[str, Any] = {**item, "pulled": False, "answered": False}
    before = set(_registered(a))
    ref = item["repo"] + (":Q4_K_M" if item["kind"] == "gguf" else "")
    started = time.monotonic()
    try:
        done = a.cli("pull", ref, timeout=3600)
    except Broken as exc:
        result["error"] = f"pull: {exc}"
        return result
    result["pull_s"] = round(time.monotonic() - started)
    added = sorted(set(_registered(a)) - before)
    if done.returncode != 0 or not added:
        result["error"] = "pull: " + (done.stdout + done.stderr).strip()[-400:]
        return result
    result["pulled"], name = True, added[0]
    try:
        # A base model (no chat template) is asked to complete, not to chat:
        # whether it runs is the question, not whether it was tuned to chat.
        shown = httpx.post(base + "/api/show", json={"model": name}, timeout=60).json()
        result["chat_template"] = bool(shown.get("template"))
        options = {"temperature": 0, "num_predict": 256}
        if result["chat_template"]:
            path = "/api/chat"
            body = {"messages": [{"role": "user", "content": QUESTION}]}
        else:
            path = "/api/generate"
            body = {"prompt": "The capital of France is", "raw": True}
        started = time.monotonic()
        out = httpx.post(
            base + path,
            json={"model": name, "stream": False, "options": options, **body},
            timeout=900,
        )
        result["chat_s"] = round(time.monotonic() - started, 1)
        is_json = out.headers.get("content-type", "").startswith("application/json")
        data = out.json() if is_json else {}
        message = data.get("message") or {"content": data.get("response", "")}
        text = (message.get("content") or "") + " " + (message.get("thinking") or "")
        result["status_code"] = out.status_code
        result["reply"] = text.strip()[:160] or out.text[:200]
        result["answered"] = out.status_code == 200 and "paris" in text.lower()
        ps = httpx.get(base + "/api/ps", timeout=30).json().get("models") or []
        mine = [m for m in ps if m.get("name") == name]
        details = (mine[0].get("details") or {}) if mine else {}
        result["engine"] = details.get("acceleration") or details.get("format")
    except (httpx.HTTPError, ValueError) as exc:
        result["error"] = f"chat: {type(exc).__name__}: {exc}"
    finally:
        httpx.post(base + "/api/generate", json={"model": name, "keep_alive": 0}, timeout=120)
        a.cli("rm", name, "--yes", timeout=300)
    return result


def _ran(result: dict) -> bool:
    """It loaded and produced text: whether a base model (no chat template)
    also knows the answer is a question about the model, not about HFL."""
    return result.get("status_code") == 200 and bool((result.get("reply") or "").strip())


def report(work: Path, results: list[dict]) -> Path:
    ok = [r for r in results if r["answered"]]
    ran = [r for r in results if _ran(r)]
    lines = [
        "# HFL compatibility sweep",
        "",
        f"{len(ran)} of {len(results)} loaded and answered; {len(ok)} said Paris "
        "(base models, without a chat template, are asked to complete a sentence).",
        "",
        "| repo | kind | size | pulled | ran | Paris | engine | pull s | chat s | note |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        size = f"{r['params_b']} B params" if "params_b" in r else f"{r.get('size_gb')} GB"
        note = (r.get("error") or ("" if r["answered"] else r.get("reply", ""))).replace("|", "/")
        lines.append(
            f"| {r['repo']} | {r['kind']} | {size} | {'yes' if r['pulled'] else 'NO'} | "
            f"{'yes' if _ran(r) else 'NO'} | {'yes' if r['answered'] else 'no'} | "
            f"{r.get('engine') or ''} | "
            f"{r.get('pull_s', '')} | {r.get('chat_s', '')} | {note[:140].replace(chr(10), ' ')} |"
        )
    path = work / "SWEEP.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--n", type=int, default=15, help="models per list")
    parser.add_argument("--max-params", type=float, default=4.0, help="billions (safetensors)")
    parser.add_argument("--max-gguf-gb", type=float, default=6.0)
    parser.add_argument("--list", action="store_true", help="list the candidates and stop")
    parser.add_argument("--only", help="comma-separated repos to re-run (results replaced)")
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    items = safetensors_candidates(args.n, args.max_params) + gguf_candidates(
        args.n, args.max_gguf_gb
    )
    if args.list:
        for item in items:
            print(json.dumps(item))
        return 0
    hfl = work / "venv" / "bin" / "hfl"
    if not hfl.exists():
        raise SystemExit(f"{hfl} does not exist: run audit/local_audit.py --setup first")
    store = work / "sweep.json"
    results: list[dict] = json.loads(store.read_text()) if store.exists() else []
    if args.only:
        wanted = {r.strip() for r in args.only.split(",")}
        items = [i for i in items if i["repo"] in wanted]
        results = [r for r in results if r["repo"] not in wanted]
    done = {r["repo"] for r in results}
    audit = Audit(hfl, work)
    try:
        with audit.server() as base:
            for item in items:
                if item["repo"] in done:
                    continue
                print(f"sweep: {item['repo']} ({item['kind']})", flush=True)
                result = try_model(audit, base, item)
                note = result.get("error") or result.get("reply", "")[:80]
                verdict = "OK" if result["answered"] else "FAIL"
                print(f"  {verdict} {result.get('engine') or ''} {note}", flush=True)
                results.append(result)
                store.write_text(json.dumps(results, indent=1))
    finally:
        audit.close()
    print(f"report: {report(work, results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
