# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl.models.capabilities.detect_capabilities``.

Capability detection is consumed by ``/api/show`` and by tooling that
decides whether to enable tool-calling, vision, embeddings or
reasoning channels. False negatives silently disable features;
false positives merely trigger graceful-degradation, so the detector
errs toward opting in.
"""

from __future__ import annotations

import pytest

from hfl.models.capabilities import detect_capabilities
from hfl.models.manifest import ModelManifest


def _manifest(**overrides) -> ModelManifest:
    """Build a minimal manifest with optional field overrides."""
    base = {
        "name": "test-model",
        "repo_id": "test/test-model",
        "local_path": "/tmp/m",
        "format": "gguf",
    }
    base.update(overrides)
    return ModelManifest(**base)


class TestCompletion:
    def test_ordinary_llm_has_completion(self):
        caps = detect_capabilities(_manifest(name="llama-3-8b", architecture="llama3"))
        assert "completion" in caps

    def test_tts_has_no_completion(self):
        caps = detect_capabilities(
            _manifest(name="bark-small", architecture="bark", model_type="tts")
        )
        assert "completion" not in caps

    def test_embedding_model_has_no_completion(self):
        """Pure embedding models do NOT advertise completion — they
        can't produce tokens, only vectors."""
        caps = detect_capabilities(_manifest(name="nomic-embed-text-v1.5", architecture="bert"))
        assert "completion" not in caps
        assert "embedding" in caps


class TestTools:
    @pytest.mark.parametrize(
        "name,arch",
        [
            ("qwen-coder:7b", "qwen"),
            ("Qwen2.5-Coder-7B-Instruct", "qwen"),
            ("meta-llama/Meta-Llama-3.1-70B", "llama3"),
            ("mistral-7b-instruct", "mistral"),
            ("gemma-4-27b-it", "gemma4"),
            ("mixtral-8x7b", "mixtral"),
        ],
    )
    def test_tool_capable_families(self, name, arch):
        caps = detect_capabilities(_manifest(name=name, architecture=arch))
        assert "tools" in caps, f"{name} should advertise tools"

    def test_random_family_no_tools(self):
        caps = detect_capabilities(_manifest(name="phi-2", architecture="phi"))
        assert "tools" not in caps


class TestInsert:
    @pytest.mark.parametrize(
        "name",
        [
            "codellama-13b",
            "CodeGemma-7B",
            "starcoder2-15b",
            "qwen-coder:7b",
            "qwen2.5-coder:32b-instruct",
            "deepseek-coder-33b",
        ],
    )
    def test_fim_families(self, name):
        caps = detect_capabilities(_manifest(name=name, architecture="unknown"))
        assert "insert" in caps, f"{name} should advertise insert"

    def test_plain_llm_no_insert(self):
        caps = detect_capabilities(_manifest(name="llama-3-8b", architecture="llama3"))
        assert "insert" not in caps


class TestVision:
    @pytest.mark.parametrize(
        "name",
        [
            "llava-v1.6-34b",
            "bakllava-v1",
            "llama-3.2-vision-11b",
            "llama-4-scout-17b-16e",
            "gemma-3-4b-it",
            "qwen2-vl-7b-instruct",
            "internvl-chat-v1-5",
            "pixtral-12b",
            "molmo-7b-d",
        ],
    )
    def test_vision_families(self, name):
        caps = detect_capabilities(_manifest(name=name, architecture="unknown"))
        assert "vision" in caps, f"{name} should advertise vision"

    def test_text_only_llm_no_vision(self):
        caps = detect_capabilities(_manifest(name="llama-3-8b-instruct", architecture="llama3"))
        assert "vision" not in caps


class TestEmbedding:
    @pytest.mark.parametrize(
        "name,arch",
        [
            ("nomic-embed-text-v1.5", "bert"),
            ("jina-embeddings-v2-base-en", "bert"),
            ("bge-m3", "bge"),
            ("mxbai-embed-large-v1", "bert"),
            ("e5-large-v2", "e5"),
            ("stella_en_400M_v5", "stella"),
            ("snowflake-arctic-embed-l", "arctic-embed"),
        ],
    )
    def test_embedding_families(self, name, arch):
        caps = detect_capabilities(_manifest(name=name, architecture=arch))
        assert "embedding" in caps
        assert "completion" not in caps

    def test_explicit_model_type_embed(self):
        """A manifest explicitly typed as ``embed`` is embedding-only,
        regardless of its architecture naming."""
        caps = detect_capabilities(
            _manifest(name="my-custom-embedder", architecture="unknown", model_type="embed")
        )
        assert "embedding" in caps
        assert "completion" not in caps


class TestThinking:
    @pytest.mark.parametrize(
        "name",
        [
            "gemma-4-27b",
            "deepseek-r1-distill-qwen-32b",
            "qwen3-thinking-30b",
            "gpt-oss-20b",
            "o1-mini",
            "o3-reasoning-preview",
        ],
    )
    def test_reasoning_families(self, name):
        caps = detect_capabilities(_manifest(name=name, architecture="unknown"))
        assert "thinking" in caps

    def test_plain_llm_no_thinking(self):
        caps = detect_capabilities(_manifest(name="llama-3-8b", architecture="llama3"))
        assert "thinking" not in caps


class TestCombined:
    def test_gemma4_tools_plus_thinking(self):
        """Gemma 4 ships native tool calls AND a reasoning channel."""
        caps = detect_capabilities(_manifest(name="gemma-4-27b-it", architecture="gemma4"))
        assert "tools" in caps
        assert "thinking" in caps
        assert "completion" in caps

    def test_qwen_coder_tools_plus_insert(self):
        """Qwen2.5-Coder is both tool-capable and FIM-capable."""
        caps = detect_capabilities(_manifest(name="qwen2.5-coder:7b-instruct", architecture="qwen"))
        assert "tools" in caps
        assert "insert" in caps
        assert "completion" in caps

    def test_gemma3_vision_plus_tools(self):
        """Gemma 3 multimodal is both vision and tool-capable."""
        caps = detect_capabilities(_manifest(name="gemma-3-12b-it", architecture="gemma"))
        assert "vision" in caps

    def test_stable_order(self):
        """Repeated detection produces identical lists (for snapshot
        tests and deterministic /api/show responses)."""
        mf = _manifest(name="qwen-coder:7b", architecture="qwen")
        assert detect_capabilities(mf) == detect_capabilities(mf)

    def test_primary_caps_first(self):
        """``completion`` (or ``embedding``) always appears first; the
        rest are alphabetical. This guarantees stable, readable UI
        rendering."""
        caps = detect_capabilities(_manifest(name="qwen2.5-coder:7b", architecture="qwen"))
        assert caps[0] in {"completion", "embedding"}
        rest = caps[1:]
        assert rest == sorted(rest)
