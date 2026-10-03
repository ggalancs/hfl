# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Coverage for ``hfl.models``: templates, thinking controls, registry
failure paths, removal and import edge cases.

jinja2 is not a core dependency (the CI venv lacks it), so the template
environment is exercised against a small fake ``jinja2`` injected in
``sys.modules``; the fake's templates are Python callables, which lets the
tests pin the *decision logic* of ``_from_template`` exactly.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfl.models import chat_template as ct
from hfl.models import thinking
from hfl.models.manifest import ModelManifest

# ----------------------------------------------------------------------
# Fake jinja2
# ----------------------------------------------------------------------


class _TemplateError(Exception):
    pass


class _Extension:
    pass


def _loopcontrols():  # stand-in object for jinja2.ext.loopcontrols
    return None


class _Env:
    renderers: dict = {}

    def __init__(self, trim_blocks=False, lstrip_blocks=False, extensions=()):
        self.options = {"trim_blocks": trim_blocks, "lstrip_blocks": lstrip_blocks}
        self.extensions = list(extensions)
        self.filters: dict = {}
        self.globals: dict = {}

    def from_string(self, source):
        if source not in self.renderers:
            raise _TemplateError(f"cannot compile {source!r}")
        return SimpleNamespace(render=self.renderers[source])


@pytest.fixture
def fake_jinja(monkeypatch):
    j = types.ModuleType("jinja2")
    j.TemplateError = _TemplateError
    ext = types.ModuleType("jinja2.ext")
    ext.Extension = _Extension
    ext.loopcontrols = _loopcontrols
    sandbox = types.ModuleType("jinja2.sandbox")
    sandbox.ImmutableSandboxedEnvironment = _Env
    j.ext, j.sandbox = ext, sandbox
    monkeypatch.setitem(sys.modules, "jinja2", j)
    monkeypatch.setitem(sys.modules, "jinja2.ext", ext)
    monkeypatch.setitem(sys.modules, "jinja2.sandbox", sandbox)
    _Env.renderers = {}
    thinking._from_template.cache_clear()
    thinking._for_file.cache_clear()
    yield _Env.renderers
    thinking._from_template.cache_clear()
    thinking._for_file.cache_clear()


class TestTemplateEnv:
    def test_env_has_what_chat_templates_expect(self, fake_jinja):
        env = ct.template_env()
        assert isinstance(env, _Env)
        assert env.options == {"trim_blocks": True, "lstrip_blocks": True}
        # tojson keeps non-ASCII and honours indent.
        assert env.filters["tojson"]({"a": "é"}) == '{"a": "é"}'
        assert env.filters["tojson"]([1], indent=2) == "[\n  1\n]"
        assert env.filters["tojson"]("é", ensure_ascii=True) == '"\\u00e9"'
        with pytest.raises(_TemplateError, match="no system role"):
            env.globals["raise_exception"]("no system role")
        assert env.globals["strftime_now"]("%Y") == datetime.now().strftime("%Y")

    def test_generation_tag_renders_its_content(self, fake_jinja):
        env = ct.template_env()
        assert env.extensions[0] is _loopcontrols
        gen_cls = env.extensions[1]
        assert issubclass(gen_cls, _Extension) and gen_cls.tags == {"generation"}

        calls = []

        class _Parser:
            stream = iter(["generation-token", "body"])

            def parse_statements(self, end_tokens, drop_needle):
                calls.append((end_tokens, drop_needle))
                return ["body-nodes"]

        parser = _Parser()
        assert gen_cls().parse(parser) == ["body-nodes"]
        assert next(parser.stream) == "body"  # the tag token was consumed
        assert calls == [(("name:endgeneration",), True)]

    def test_without_jinja2_there_is_no_env(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "jinja2", None)
        assert ct.template_env() is None


class TestModelTemplate:
    def test_repair_fixes_doubled_braces_only(self):
        wrong = 'call {{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}}'
        fixed = ct.repair_chat_template(wrong)
        assert fixed == 'call {\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}'
        assert ct.repair_chat_template("{{ messages }}") == "{{ messages }}"

    def test_own_template_wins(self, tmp_path):
        (tmp_path / "chat_template.jinja").write_text("from file")
        m = SimpleNamespace(chat_template="mine", local_path=str(tmp_path))
        assert ct.model_template(m) == "mine"

    def test_jinja_file_then_tokenizer_config(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "cfg"}))
        m = SimpleNamespace(chat_template=None, local_path=str(tmp_path))
        assert ct.model_template(m) == "cfg"
        (tmp_path / "chat_template.jinja").write_text("jinja", encoding="utf-8")
        assert ct.model_template(m) == "jinja"

    @pytest.mark.parametrize(
        "value, expected",
        [
            ([{"name": "tool_use", "template": "T"}, {"name": "default", "template": "D"}], "D"),
            ([{"name": "tool_use", "template": "T"}, "junk"], ""),
            ([{"name": "default", "template": 3}], ""),
            (None, ""),
            (42, ""),
        ],
    )
    def test_tokenizer_config_shapes(self, tmp_path, value, expected):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": value}))
        m = SimpleNamespace(local_path=str(tmp_path))
        assert ct.model_template(m) == expected

    def test_unreadable_config_gives_empty(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text("{not json")
        assert ct.model_template(SimpleNamespace(local_path=str(tmp_path))) == ""

    def test_no_path_and_folder_without_files(self, tmp_path):
        assert ct.model_template(SimpleNamespace()) == ""
        assert ct.model_template(SimpleNamespace(local_path=str(tmp_path))) == ""


# ----------------------------------------------------------------------
# thinking: decision logic over rendered outputs
# ----------------------------------------------------------------------


def _effort(default="medium"):
    def render(reasoning_effort=None, **_):
        return f"effort={reasoning_effort or default}"

    return render


def _switch(variable, default_on):
    def render(**kw):
        value = kw.get(variable)
        on = default_on if value is None else value
        return "think" if on else "no-think"

    return render


class TestFromTemplate:
    def test_reasoning_effort_levels_and_default(self, fake_jinja):
        fake_jinja["gpt-oss"] = _effort("medium")
        assert thinking._from_template("gpt-oss") == {
            "values": ["low", "medium", "high"],
            "default": "medium",
        }

    def test_levels_without_a_matching_default_fall_through(self, fake_jinja):
        fake_jinja["odd"] = _effort("extreme")  # unset matches no level
        assert thinking._from_template("odd") is None

    def test_enable_thinking_default_on_and_off(self, fake_jinja):
        fake_jinja["qwen3"] = _switch("enable_thinking", default_on=True)
        fake_jinja["gemma4"] = _switch("enable_thinking", default_on=False)
        assert thinking._from_template("qwen3") == {"values": [False, True], "default": True}
        assert thinking._from_template("gemma4") == {"values": [False, True], "default": False}

    def test_thinking_variable(self, fake_jinja):
        fake_jinja["dsv31"] = _switch("thinking", default_on=False)
        assert thinking._from_template("dsv31") == {"values": [False, True], "default": False}

    def test_switch_whose_unset_output_is_neither(self, fake_jinja):
        def render(enable_thinking=None, **_):
            return {True: "a", False: "b", None: "c"}[enable_thinking]

        fake_jinja["weird"] = render
        assert thinking._from_template("weird") is None

    def test_variable_that_fails_to_render_is_skipped(self, fake_jinja):
        def render(enable_thinking=None, **_):
            if enable_thinking is False:
                raise ValueError("boom")
            return "x"

        fake_jinja["half"] = render
        assert thinking._from_template("half") is None

    def test_unrenderable_uncompilable_and_empty(self, fake_jinja):
        def broken(**_):
            raise RuntimeError("undefined")

        fake_jinja["broken"] = broken
        assert thinking._from_template("broken") is None
        assert thinking._from_template("not registered") is None
        assert thinking._from_template("") is None

    def test_render_gets_one_user_turn_and_generation_prompt(self, fake_jinja):
        seen = {}

        def render(**kw):
            seen.update(kw)
            return "same"

        fake_jinja["spy"] = render
        assert thinking._from_template("spy") is None  # nothing changes output
        assert seen["messages"] == [{"role": "user", "content": "hi"}]
        assert seen["add_generation_prompt"] is True
        assert seen["bos_token"] == "" and seen["eos_token"] == ""


class TestThinkingControls:
    def test_template_answer_is_returned(self, fake_jinja, tmp_path):
        fake_jinja["qwen3"] = _switch("enable_thinking", default_on=True)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        m = SimpleNamespace(local_path=str(model), chat_template="qwen3", name="qwen3")
        assert thinking.thinking_controls(m) == {"values": [False, True], "default": True}

    def test_missing_file_still_uses_own_template(self, fake_jinja, tmp_path):
        fake_jinja["gpt"] = _effort("low")
        m = SimpleNamespace(local_path=str(tmp_path / "gone.gguf"), chat_template="gpt")
        assert thinking.thinking_controls(m)["default"] == "low"

    def test_reasoning_model_without_switch_always_thinks(self, fake_jinja, monkeypatch):
        monkeypatch.setattr(thinking, "detect_capabilities", lambda m: ["completion", "thinking"])
        m = SimpleNamespace(local_path="", chat_template=None)
        assert thinking.thinking_controls(m) == {"values": [True], "default": True}

    def test_capability_detection_failure_means_none(self, fake_jinja, monkeypatch):
        def _boom(m):
            raise RuntimeError("bad manifest")

        monkeypatch.setattr(thinking, "detect_capabilities", _boom)
        m = SimpleNamespace(local_path="", chat_template=123)  # non-str template ignored
        assert thinking.thinking_controls(m) is None

    def test_reasoning_model_without_jinja_is_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "jinja2", None)
        thinking._for_file.cache_clear()
        thinking._from_template.cache_clear()
        monkeypatch.setattr(thinking, "detect_capabilities", lambda m: ["thinking"])
        assert thinking.thinking_controls(SimpleNamespace(local_path="")) is None
        thinking._for_file.cache_clear()


# ----------------------------------------------------------------------
# Registry failure paths
# ----------------------------------------------------------------------


def _manifest(name, path="/tmp/x", **kw):
    return ModelManifest(name=name, repo_id=f"org/{name}", local_path=path, format="gguf", **kw)


@pytest.fixture
def registry_mod(temp_config):
    from hfl.models import registry

    return registry


class TestRegistryLocking:
    def test_lock_retries_until_it_gets_the_lock(self, registry_mod, monkeypatch):
        attempts = []

        def flaky(fd, exclusive):
            attempts.append(exclusive)
            if len(attempts) < 3:
                raise BlockingIOError("busy")

        monkeypatch.setattr(registry_mod, "_lock_file", flaky)
        sleeps = []
        monkeypatch.setattr(registry_mod.time, "sleep", sleeps.append)
        reg = registry_mod.ModelRegistry()
        with reg._file_lock(exclusive=True):
            pass
        assert attempts == [True, True, True]
        assert sleeps == [0.1, 0.1]

    def test_lock_gives_up_after_ten_tries_and_unlocks(self, registry_mod, monkeypatch):
        def never(fd, exclusive):
            raise OSError("locked")

        unlocked = []
        monkeypatch.setattr(registry_mod, "_lock_file", never)
        monkeypatch.setattr(registry_mod, "_unlock_file", unlocked.append)
        monkeypatch.setattr(registry_mod.time, "sleep", lambda s: None)
        reg = registry_mod.ModelRegistry()
        with pytest.raises(OSError, match="locked"):
            with reg._file_lock():
                pytest.fail("body must not run without the lock")
        assert len(unlocked) == 1


class TestRegistryRecovery:
    def test_non_list_file_recovers_from_backup(self, registry_mod, temp_config):
        path = temp_config.registry_path
        backup = path.with_suffix(".json.bak")
        backup.write_text(json.dumps([_manifest("saved").to_dict()]))
        path.write_text(json.dumps({"not": "a list"}))
        reg = registry_mod.ModelRegistry()
        assert [m.name for m in reg.list_all()] == ["saved"]
        # The backup was restored as the main file.
        assert json.loads(path.read_text())[0]["name"] == "saved"

    def test_corrupt_backup_too_gives_empty_and_event(self, registry_mod, temp_config, monkeypatch):
        events = []
        import hfl.events

        monkeypatch.setattr(hfl.events, "emit", lambda *a, **kw: events.append(kw))
        path = temp_config.registry_path
        path.write_text("{broken")
        path.with_suffix(".json.bak").write_text("{also broken")
        reg = registry_mod.ModelRegistry()
        assert len(reg) == 0
        assert events[-1]["recovery_status"] == "recovery_failed"
        assert events[-1]["error_message"]

    def test_events_module_missing_is_tolerated(self, registry_mod, temp_config, monkeypatch):
        monkeypatch.setitem(sys.modules, "hfl.events", None)
        temp_config.registry_path.write_text("{broken")
        assert len(registry_mod.ModelRegistry()) == 0


class TestRegistrySave:
    def test_backup_failure_does_not_block_save(self, registry_mod, monkeypatch, caplog):
        reg = registry_mod.ModelRegistry()
        reg.add(_manifest("a"))  # file exists now

        def no_copy(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(registry_mod.shutil, "copy2", no_copy)
        with caplog.at_level(logging.WARNING, logger="hfl.models.registry"):
            reg.add(_manifest("b"))
        assert "Failed to create backup" in caplog.text
        assert {m.name for m in registry_mod.ModelRegistry().list_all()} == {"a", "b"}

    def test_failed_write_cleans_temp_and_raises(self, registry_mod, temp_config, monkeypatch):
        reg = registry_mod.ModelRegistry()
        reg.add(_manifest("a"))
        before = temp_config.registry_path.read_text()

        def no_replace(self, target):
            raise OSError("rename refused")

        monkeypatch.setattr(Path, "replace", no_replace)
        with pytest.raises(OSError, match="rename refused"):
            reg.add(_manifest("b"))
        monkeypatch.undo()
        assert not temp_config.registry_path.with_suffix(".json.tmp").exists()
        assert temp_config.registry_path.read_text() == before


class TestRegistryQueries:
    def test_validate_reports_duplicate_aliases_and_missing_paths(self, registry_mod, tmp_path):
        reg = registry_mod.ModelRegistry()
        reg._models = [
            _manifest("a", path=str(tmp_path), alias="same"),
            _manifest("b", path="", alias="same"),
        ]
        reg._indexes_dirty = True
        ok, errors = reg.validate_integrity()
        assert ok is False
        assert any("Duplicate aliases" in e and "same" in e for e in errors)
        assert "Model 'b' has no local_path" in errors
        assert not any("'a'" in e for e in errors)

    def test_find_pulled_skips_other_repos(self, registry_mod):
        reg = registry_mod.ModelRegistry()
        reg.add(_manifest("wanted"))
        reg.add(_manifest("other"))
        assert reg.find_pulled("org/wanted").name == "wanted"
        assert reg.find_pulled("org/none") is None

    def test_find_pulled_matches_quantization_case_insensitively(self, registry_mod):
        reg = registry_mod.ModelRegistry()
        reg.add(_manifest("m-q8", quantization="Q8_0"))
        assert reg.find_pulled("ORG/m-q8", "q8_0").name == "m-q8"
        assert reg.find_pulled("org/m-q8", "Q4_K_M") is None
        assert reg.find_pulled("org/m-q8").name == "m-q8"


# ----------------------------------------------------------------------
# Removal, import, manifest, provenance edges
# ----------------------------------------------------------------------


class TestRemovalEdges:
    def test_failed_restore_still_reports_in_use(self, tmp_path, monkeypatch):
        from hfl.models import removal

        target = tmp_path / "m.gguf"
        target.write_bytes(b"x")
        real_rename = Path.rename
        renames = []

        def rename(self, dest):
            renames.append(Path(dest).name)
            if len(renames) == 1:
                return real_rename(self, dest)
            raise OSError("cannot restore")

        def unlink(self, missing_ok=False):
            raise PermissionError("open elsewhere")

        monkeypatch.setattr(Path, "rename", rename)
        monkeypatch.setattr(Path, "unlink", unlink)
        with pytest.raises(removal.ModelInUse):
            removal._delete(target)
        monkeypatch.undo()
        assert renames == [renames[0], "m.gguf"]  # moved aside, then tried back
        assert not target.exists()
        assert len(list(tmp_path.glob("m.gguf.removing-*"))) == 1

    def test_drop_download_folder_ignores_missing_folder(self, tmp_path):
        from hfl.models import removal

        models = tmp_path / "models"
        models.mkdir()
        removal._drop_download_folder(models / "gone", models)  # no raise
        assert list(models.iterdir()) == []

    def test_resolved_falls_back_on_oserror(self, monkeypatch):
        from hfl.models import removal

        def boom(self, strict=False):
            raise OSError("loop")

        monkeypatch.setattr(Path, "resolve", boom)
        p = Path("some/where")
        assert removal._resolved(p) is p

    def test_missing_file_inside_models_dir_drops_only_entry(self, temp_config):
        from hfl.models.registry import ModelRegistry
        from hfl.models.removal import remove_model

        reg = ModelRegistry()
        gone = temp_config.models_dir / "org" / "gone.gguf"
        m = _manifest("gone", path=str(gone))
        reg.add(m)
        result = remove_model(reg, m)
        assert result.deleted is False and result.kept_outside is None
        assert reg.get("gone") is None


class TestImporterEdges:
    def test_unreadable_path_is_not_gguf(self, tmp_path):
        from hfl.models.importer import _is_gguf

        assert _is_gguf(tmp_path / "missing.gguf") is False
        (tmp_path / "ok.gguf").write_bytes(b"GGUF\x03")
        assert _is_gguf(tmp_path / "ok.gguf") is True

    def test_config_that_is_not_an_object_is_refused(self, tmp_path):
        from hfl.models.importer import ImportRefused, manifest_for_folder

        (tmp_path / "config.json").write_text("[1, 2]")
        with pytest.raises(ImportRefused) as exc:
            manifest_for_folder(tmp_path, "x")
        assert exc.value.key == "import.bad_config"
        assert exc.value.fields == {"path": str(tmp_path)}

    def test_config_with_no_telling_task_is_a_chat_model(self, tmp_path, monkeypatch):
        from hfl.converter import formats
        from hfl.models.importer import manifest_for_folder

        (tmp_path / "config.json").write_text('{"torch_dtype": "bfloat16"}')
        monkeypatch.setattr(formats, "detect_model_type", lambda folder: formats.ModelType.UNKNOWN)
        m = manifest_for_folder(tmp_path, "x")
        assert m.name == "x" and m.model_type == "llm"

    def test_unsupported_model_type_is_refused(self, tmp_path, monkeypatch):
        from hfl.converter import formats
        from hfl.models.importer import ImportRefused, manifest_for_folder

        (tmp_path / "config.json").write_text("{}")
        monkeypatch.setattr(formats, "detect_model_type", lambda folder: formats.ModelType.STT)
        with pytest.raises(ImportRefused) as exc:
            manifest_for_folder(tmp_path, "x")
        assert exc.value.key == "import.unsupported"
        assert exc.value.fields["kind"] == formats.get_model_type_display_name(
            formats.ModelType.STT
        )


class TestManifestAndProvenanceEdges:
    def test_hash_that_cannot_be_computed_fails_verification(self, tmp_path, monkeypatch):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"abc")
        m = _manifest("m", path=str(f), size_bytes=3, file_hash="00")
        monkeypatch.setattr(ModelManifest, "compute_hash", lambda self: None)
        assert m.verify_integrity() == (False, "Failed to compute file hash")

    def test_provenance_file_that_is_not_a_list_is_kept_aside(self, tmp_path):
        from hfl.models.provenance import ProvenanceLog

        log = tmp_path / "provenance.json"
        log.write_text(json.dumps({"records": []}))
        p = ProvenanceLog(log_path=log)
        assert p.get_all() == []
        aside = list(tmp_path.glob("provenance.json.corrupt-*"))
        assert len(aside) == 1
        assert json.loads(aside[0].read_text()) == {"records": []}

    def test_capabilities_without_name(self):
        from hfl.models.capabilities import detect_capabilities

        m = _manifest("x")
        m.name, m.repo_id, m.architecture = "", "deepseek-ai/DeepSeek-R1", None
        assert "thinking" in detect_capabilities(m)

    def test_capabilities_from_name_only(self):
        from hfl.models.capabilities import detect_capabilities

        m = _manifest("deepseek-r1-distill")
        m.repo_id, m.architecture = "", None
        assert "thinking" in detect_capabilities(m)
