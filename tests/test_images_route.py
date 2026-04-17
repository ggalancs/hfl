# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the image-generation engine + route (Phase 16 — V2 row 17)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.engine import diffusers_engine
from hfl.engine.diffusers_engine import DEFAULT_SIZE, DEFAULT_STEPS, DiffusersEngine, ImageResult


class TestConstants:
    def test_defaults_are_sensible(self):
        assert DEFAULT_SIZE in (512, 768, 1024)
        assert DEFAULT_STEPS >= 20


class TestLoadGate:
    def test_load_raises_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(diffusers_engine, "is_available", lambda: False)
        engine = DiffusersEngine()
        with pytest.raises(RuntimeError):
            engine.load("stable-diffusion")


class TestRoute:
    def test_501_when_no_backend(self, monkeypatch, temp_config):
        from hfl.api import routes_images

        monkeypatch.setattr(routes_images, "is_available", lambda: False)
        client = TestClient(app)
        resp = client.post(
            "/api/images/generate",
            json={"model": "stable", "prompt": "a cat"},
        )
        assert resp.status_code == 501

    def test_happy_path(self, monkeypatch, temp_config):
        from hfl.api import routes_images

        monkeypatch.setattr(routes_images, "is_available", lambda: True)

        class _FakeEngine:
            def __init__(self):
                self._loaded = False

            def load(self, model, device=None):
                self._loaded = True
                self.model = model

            def unload(self):
                self._loaded = False

            @property
            def is_loaded(self):
                return self._loaded

            def generate(
                self,
                prompt,
                *,
                negative_prompt=None,
                width=1024,
                height=1024,
                steps=30,
                guidance_scale=7.5,
                seed=None,
            ):
                return ImageResult(
                    image_png_base64="AAAA",
                    seed=seed,
                    duration_s=0.1,
                    width=width,
                    height=height,
                )

        monkeypatch.setattr(routes_images, "DiffusersEngine", lambda: _FakeEngine())

        client = TestClient(app)
        resp = client.post(
            "/api/images/generate",
            json={
                "model": "sdxl",
                "prompt": "a cat on a rocket",
                "size": "512x512",
                "steps": 20,
                "seed": 42,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["image"]["format"] == "png"
        assert body["image"]["b64"] == "AAAA"
        assert body["width"] == 512
        assert body["height"] == 512

    def test_bad_size_rejected(self, monkeypatch, temp_config):
        from hfl.api import routes_images

        monkeypatch.setattr(routes_images, "is_available", lambda: True)
        client = TestClient(app)
        resp = client.post(
            "/api/images/generate",
            json={"model": "x", "prompt": "x", "size": "bad"},
        )
        assert resp.status_code == 422

    def test_empty_prompt_rejected(self, monkeypatch, temp_config):
        from hfl.api import routes_images

        monkeypatch.setattr(routes_images, "is_available", lambda: True)
        client = TestClient(app)
        resp = client.post(
            "/api/images/generate",
            json={"model": "x", "prompt": ""},
        )
        assert resp.status_code == 422
