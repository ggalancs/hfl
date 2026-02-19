# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the CLI module (main commands)."""

import pytest
import sys
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner


@pytest.fixture
def mock_llama_cpp():
    """Mock llama_cpp for tests that need it."""
    mock = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": mock}):
        yield mock


@pytest.fixture
def runner():
    """Runner for CLI tests."""
    return CliRunner()


@pytest.fixture
def cli_app():
    """CLI application for tests."""
    from hfl.cli.main import app
    return app


class TestVersionCommand:
    """Tests for version command."""

    def test_version(self, runner, cli_app):
        """Verifies version command."""
        result = runner.invoke(cli_app, ["version"])

        assert result.exit_code == 0
        assert "hfl" in result.stdout
        assert "0.1.0" in result.stdout
        assert "HRUL" in result.stdout  # License reference


class TestListCommand:
    """Tests for list command."""

    def test_list_empty(self, runner, cli_app, temp_config):
        """Empty model list."""
        result = runner.invoke(cli_app, ["list"])

        assert result.exit_code == 0
        assert "No downloaded models" in result.stdout

    def test_list_with_models(self, runner, cli_app, populated_registry):
        """List with models."""
        result = runner.invoke(cli_app, ["list"])

        assert result.exit_code == 0
        # Rich table truncates long names with "..." (U+2026)
        # We verify prefixes or the quantization that always appears
        assert "Q4_K_M" in result.stdout
        assert "Q5_K_M" in result.stdout
        assert "test-org" in result.stdout or "test-model" in result.stdout
        assert "other-org" in result.stdout or "another" in result.stdout


class TestInspectCommand:
    """Tests for inspect command."""

    def test_inspect_not_found(self, runner, cli_app, temp_config):
        """Inspect non-existent model."""
        result = runner.invoke(cli_app, ["inspect", "nonexistent"])

        assert result.exit_code == 1
        assert "Model not found" in result.stdout

    def test_inspect_success(self, runner, cli_app, populated_registry):
        """Inspect existing model."""
        result = runner.invoke(cli_app, ["inspect", "test-model-q4_k_m"])

        assert result.exit_code == 0
        assert "test-model-q4_k_m" in result.stdout
        assert "test-org/test-model" in result.stdout
        assert "gguf" in result.stdout.lower()


class TestAliasCommand:
    """Tests for alias command."""

    def test_alias_model_not_found(self, runner, cli_app, temp_config):
        """Alias to non-existent model."""
        result = runner.invoke(cli_app, ["alias", "nonexistent", "myalias"])

        assert result.exit_code == 1
        assert "Model not found" in result.stdout

    def test_alias_success(self, runner, cli_app, temp_config, temp_dir):
        """Successfully assign alias."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="very-long-model-name-q4_k_m",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        result = runner.invoke(cli_app, ["alias", "very-long-model-name-q4_k_m", "short"])

        assert result.exit_code == 0
        assert "Alias assigned" in result.stdout
        assert "short" in result.stdout

        # Verify that the alias works
        registry = ModelRegistry()
        model = registry.get("short")
        assert model is not None
        assert model.name == "very-long-model-name-q4_k_m"

    def test_alias_duplicate_rejected(self, runner, cli_app, temp_config, temp_dir):
        """Duplicate alias is rejected."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="model-1",
            repo_id="test-org/model-1",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            alias="taken",
        ))
        registry.add(ModelManifest(
            name="model-2",
            repo_id="test-org/model-2",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
        ))

        result = runner.invoke(cli_app, ["alias", "model-2", "taken"])

        assert result.exit_code == 1
        assert "already in use" in result.stdout

    def test_run_by_alias(self, runner, cli_app, temp_config, temp_dir):
        """Run model by alias."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="very-long-model-name",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
            alias="short",
        ))

        with patch("hfl.engine.selector.select_engine") as mock_select:
            mock_engine = MagicMock()
            mock_engine.chat_stream.return_value = iter(["Hello"])
            mock_select.return_value = mock_engine

            result = runner.invoke(cli_app, ["run", "short"], input="/exit\n")

            assert "Loading" in result.stdout

    def test_inspect_by_alias(self, runner, cli_app, temp_config, temp_dir):
        """Inspect model by alias."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="long-name",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            alias="mymodel",
        ))

        result = runner.invoke(cli_app, ["inspect", "mymodel"])

        assert result.exit_code == 0
        assert "long-name" in result.stdout
        assert "mymodel" in result.stdout


class TestRmCommand:
    """Tests for rm command."""

    def test_rm_not_found(self, runner, cli_app, temp_config):
        """Delete non-existent model."""
        result = runner.invoke(cli_app, ["rm", "nonexistent"])

        assert result.exit_code == 1
        assert "Model not found" in result.stdout

    def test_rm_cancelled(self, runner, cli_app, populated_registry):
        """Cancel deletion."""
        result = runner.invoke(cli_app, ["rm", "test-model-q4_k_m"], input="n\n")

        assert result.exit_code == 0
        # The model should still exist
        from hfl.models.registry import ModelRegistry
        registry = ModelRegistry()
        assert registry.get("test-model-q4_k_m") is not None

    def test_rm_confirmed(self, runner, cli_app, temp_config, temp_dir):
        """Confirm deletion."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        # Create temporary file for the model
        model_path = temp_dir / "test-model"
        model_path.mkdir()
        (model_path / "model.gguf").write_bytes(b"GGUF")

        # Register model with real path
        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="rm-test-model",
            repo_id="test/rm-model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        result = runner.invoke(cli_app, ["rm", "rm-test-model"], input="y\n")

        assert result.exit_code == 0
        assert "Deleted" in result.stdout

        # Verify it was removed from the registry
        registry = ModelRegistry()
        assert registry.get("rm-test-model") is None


class TestPullCommand:
    """Tests for pull command."""

    def test_pull_success(self, runner, cli_app, temp_config):
        """Successful GGUF model pull."""
        from hfl.hub.resolver import ResolvedModel

        with patch("hfl.hub.resolver.resolve") as mock_resolve:
            with patch("hfl.hub.downloader.pull_model") as mock_pull:
                mock_resolve.return_value = ResolvedModel(
                    repo_id="test/model",
                    filename="model-Q4_K_M.gguf",
                    format="gguf",
                    quantization="Q4_K_M",
                )

                model_path = temp_config.models_dir / "test--model" / "model-Q4_K_M.gguf"
                model_path.parent.mkdir(parents=True)
                model_path.write_bytes(b"GGUF content")
                mock_pull.return_value = model_path

                # Use --skip-license to avoid HuggingFace API calls
                result = runner.invoke(cli_app, ["pull", "test/model", "--skip-license"])

                assert result.exit_code == 0
                assert "Resolving" in result.stdout

    def test_pull_with_quantize_option(self, runner, cli_app, temp_config):
        """Pull with quantization option."""
        from hfl.hub.resolver import ResolvedModel

        with patch("hfl.hub.resolver.resolve") as mock_resolve:
            with patch("hfl.hub.downloader.pull_model") as mock_pull:
                mock_resolve.return_value = ResolvedModel(
                    repo_id="test/model",
                    filename="model-Q5_K_M.gguf",
                    format="gguf",
                    quantization="Q5_K_M",
                )

                model_path = temp_config.models_dir / "model.gguf"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(b"GGUF")
                mock_pull.return_value = model_path

                # Use --skip-license to avoid HuggingFace API calls
                result = runner.invoke(cli_app, ["pull", "test/model", "-q", "Q5_K_M", "--skip-license"])

                mock_resolve.assert_called_once()

    def test_pull_invalid_format_shows_helpful_error(self, runner, cli_app, temp_config):
        """Pull with invalid format shows friendly error."""
        from huggingface_hub.utils import HFValidationError

        with patch("hfl.hub.resolver.resolve") as mock_resolve:
            mock_resolve.side_effect = HFValidationError(
                "Repo id must use alphanumeric chars, '-', '_' or '.'"
            )

            result = runner.invoke(cli_app, ["pull", "invalid:model:format", "--skip-license"])

            assert result.exit_code == 1
            # Verify that it shows error information
            assert "alphanumeric" in result.stdout or "Error" in result.stdout

    def test_pull_model_not_found_error(self, runner, cli_app, temp_config):
        """Pull with model not found shows clear error."""
        with patch("hfl.hub.resolver.resolve") as mock_resolve:
            mock_resolve.side_effect = ValueError("Model not found: nonexistent")

            result = runner.invoke(cli_app, ["pull", "nonexistent", "--skip-license"])

            assert result.exit_code == 1
            assert "Model not found" in result.stdout

    def test_pull_with_colon_quantization(self, runner, cli_app, temp_config):
        """Pull with repo:quantization format (Ollama style)."""
        from hfl.hub.resolver import ResolvedModel

        with patch("hfl.hub.resolver.resolve") as mock_resolve:
            with patch("hfl.hub.downloader.pull_model") as mock_pull:
                mock_resolve.return_value = ResolvedModel(
                    repo_id="org/model",
                    filename="model-Q4_K_M.gguf",
                    format="gguf",
                    quantization="Q4_K_M",
                )

                model_path = temp_config.models_dir / "model.gguf"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(b"GGUF")
                mock_pull.return_value = model_path

                result = runner.invoke(cli_app, ["pull", "org/model:Q4_K_M", "--skip-license"])

                assert result.exit_code == 0


class TestRunCommand:
    """Tests for run command."""

    def test_run_model_not_found(self, runner, cli_app, temp_config):
        """Run with non-existent model."""
        result = runner.invoke(cli_app, ["run", "nonexistent"])

        assert result.exit_code == 1
        assert "Model not found" in result.stdout

    def test_run_missing_dependency_shows_helpful_error(self, runner, cli_app, temp_config, temp_dir):
        """Run shows helpful error when dependency is missing."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest
        from hfl.engine.selector import MissingDependencyError

        # Create model file
        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        # Register model
        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="missing-dep-test",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        # Simulate that llama_cpp is missing (patch in selector where it's imported)
        with patch("hfl.engine.selector.select_engine") as mock_select:
            mock_select.side_effect = MissingDependencyError(
                "The llama-cpp backend requires 'llama-cpp-python'.\n"
                "Install it with: pip install llama-cpp-python"
            )

            result = runner.invoke(cli_app, ["run", "missing-dep-test"])

            assert result.exit_code == 1
            assert "Missing dependency" in result.stdout
            assert "llama-cpp-python" in result.stdout
            assert "pip install" in result.stdout

    def test_run_with_exit(self, runner, cli_app, temp_config, temp_dir):
        """Run with immediate exit."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        # Create model file
        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        # Register model
        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="run-test-model",
            repo_id="test-org/run-model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        with patch("hfl.engine.selector.select_engine") as mock_select:
            mock_engine = MagicMock()
            mock_engine.chat_stream.return_value = iter(["Hello"])
            mock_select.return_value = mock_engine

            result = runner.invoke(cli_app, ["run", "run-test-model"], input="/exit\n")

            assert "Loading" in result.stdout

    def test_run_streaming_with_special_characters(self, runner, cli_app, temp_config, temp_dir):
        """Run with special characters in response (avoids Rich markup errors)."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="special-char-model",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        with patch("hfl.engine.selector.select_engine") as mock_select:
            mock_engine = MagicMock()
            # Response with characters that could cause Rich markup errors
            mock_engine.chat_stream.return_value = iter([
                "Here's code: ",
                "[bold]",  # Could be interpreted as Rich markup
                " and ",
                "[/]",     # Could cause MarkupError
                " brackets",
            ])
            mock_select.return_value = mock_engine

            result = runner.invoke(cli_app, ["run", "special-char-model"], input="test\n/exit\n")

            # Should not have markup errors
            assert result.exit_code == 0
            assert "MarkupError" not in result.stdout
            assert "Traceback" not in result.stdout

    def test_run_streaming_complete_response(self, runner, cli_app, temp_config, temp_dir):
        """Run captures complete streaming response."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="stream-test-model",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        with patch("hfl.engine.selector.select_engine") as mock_select:
            mock_engine = MagicMock()
            mock_engine.chat_stream.return_value = iter(["Hello", " ", "World", "!"])
            mock_select.return_value = mock_engine

            result = runner.invoke(cli_app, ["run", "stream-test-model"], input="Hi\n/exit\n")

            assert result.exit_code == 0
            assert "Hello" in result.stdout
            assert "World" in result.stdout

    def test_run_empty_response(self, runner, cli_app, temp_config, temp_dir):
        """Run handles empty response correctly."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="empty-response-model",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        with patch("hfl.engine.selector.select_engine") as mock_select:
            mock_engine = MagicMock()
            mock_engine.chat_stream.return_value = iter([])  # Empty response
            mock_select.return_value = mock_engine

            result = runner.invoke(cli_app, ["run", "empty-response-model"], input="Hi\n/exit\n")

            assert result.exit_code == 0
            assert "Traceback" not in result.stdout

    def test_run_multi_turn_conversation(self, runner, cli_app, temp_config, temp_dir):
        """Run handles multi-turn conversation."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="multi-turn-model",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        with patch("hfl.engine.selector.select_engine") as mock_select:
            mock_engine = MagicMock()
            # Simulate multiple responses
            mock_engine.chat_stream.side_effect = [
                iter(["First response"]),
                iter(["Second response"]),
            ]
            mock_select.return_value = mock_engine

            result = runner.invoke(
                cli_app,
                ["run", "multi-turn-model"],
                input="Hello\nHow are you?\n/exit\n"
            )

            assert result.exit_code == 0
            assert "First response" in result.stdout
            assert "Second response" in result.stdout

    def test_run_quit_commands(self, runner, cli_app, temp_config, temp_dir):
        """Run recognizes various exit commands."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()

        for i, exit_cmd in enumerate(["/exit", "/quit", "/bye"]):
            registry.add(ModelManifest(
                name=f"quit-test-{i}",
                repo_id="test-org/model",
                local_path=str(model_path),
                format="gguf",
                size_bytes=100,
                quantization="Q4_K_M",
            ))

            with patch("hfl.engine.selector.select_engine") as mock_select:
                mock_engine = MagicMock()
                mock_select.return_value = mock_engine

                result = runner.invoke(cli_app, ["run", f"quit-test-{i}"], input=f"{exit_cmd}\n")

                assert result.exit_code == 0, f"Failed with command {exit_cmd}"
                assert "Session ended" in result.stdout

    def test_run_empty_input_ignored(self, runner, cli_app, temp_config, temp_dir):
        """Run ignores empty inputs."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="empty-input-model",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        with patch("hfl.engine.selector.select_engine") as mock_select:
            mock_engine = MagicMock()
            mock_engine.chat_stream.return_value = iter(["Response"])
            mock_select.return_value = mock_engine

            # Send empty lines that should be ignored
            result = runner.invoke(
                cli_app,
                ["run", "empty-input-model"],
                input="\n\n\nHello\n/exit\n"
            )

            assert result.exit_code == 0
            # chat_stream should only be called once (for "Hello")
            assert mock_engine.chat_stream.call_count == 1

    def test_run_with_system_prompt(self, runner, cli_app, temp_config, temp_dir):
        """Run with system prompt."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="system-prompt-model",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        with patch("hfl.engine.selector.select_engine") as mock_select:
            mock_engine = MagicMock()
            mock_engine.chat_stream.return_value = iter(["Hola"])
            mock_select.return_value = mock_engine

            result = runner.invoke(
                cli_app,
                ["run", "system-prompt-model", "--system", "Responde en español"],
                input="Hello\n/exit\n"
            )

            assert result.exit_code == 0
            # Verify that it was called with a system message
            call_args = mock_engine.chat_stream.call_args
            messages = call_args[0][0]
            assert any(m.role == "system" for m in messages)

    def test_run_unicode_response(self, runner, cli_app, temp_config, temp_dir):
        """Run handles Unicode responses correctly."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="unicode-model",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        with patch("hfl.engine.selector.select_engine") as mock_select:
            mock_engine = MagicMock()
            # Response with emojis and special characters
            mock_engine.chat_stream.return_value = iter([
                "¡Hola! ", "こんにちは ", "🎉", " Привет"
            ])
            mock_select.return_value = mock_engine

            result = runner.invoke(cli_app, ["run", "unicode-model"], input="Hi\n/exit\n")

            assert result.exit_code == 0
            assert "Traceback" not in result.stdout


class TestServeCommand:
    """Tests for serve command."""

    def test_serve_imports(self, cli_app):
        """Verifies that serve imports correctly."""
        from hfl.cli.main import serve
        assert callable(serve)


class TestSearchCommand:
    """Tests for search command."""

    def test_search_too_short(self, runner, cli_app, temp_config):
        """Search with text too short."""
        result = runner.invoke(cli_app, ["search", "ab"])

        assert result.exit_code == 1
        assert "at least 3 characters" in result.stdout

    def test_search_no_results(self, runner, cli_app, temp_config):
        """Search with no results."""
        with patch("huggingface_hub.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.list_models.return_value = []
            mock_api_class.return_value = mock_api

            result = runner.invoke(cli_app, ["search", "xyznonexistent"])

            assert result.exit_code == 0
            assert "No models found" in result.stdout

    def test_search_with_results(self, runner, cli_app, temp_config):
        """Search with results."""
        with patch("huggingface_hub.HfApi") as mock_api_class:
            mock_model = MagicMock()
            mock_model.id = "org/test-model"
            mock_model.downloads = 1000
            mock_model.likes = 50
            mock_model.siblings = []
            mock_model.pipeline_tag = "text-generation"

            mock_api = MagicMock()
            mock_api.list_models.return_value = [mock_model]
            mock_api_class.return_value = mock_api

            result = runner.invoke(cli_app, ["search", "test", "--page-size", "1"], input="q")

            assert "org/test-model" in result.stdout

    def test_search_gguf_only(self, runner, cli_app, temp_config):
        """Search GGUF only."""
        with patch("huggingface_hub.HfApi") as mock_api_class:
            mock_model_gguf = MagicMock()
            mock_model_gguf.id = "org/model-gguf"
            mock_model_gguf.downloads = 1000
            mock_model_gguf.likes = 50
            mock_model_gguf.siblings = [MagicMock(rfilename="model.gguf")]
            mock_model_gguf.pipeline_tag = "text-generation"

            mock_api = MagicMock()
            mock_api.list_models.return_value = [mock_model_gguf]
            mock_api_class.return_value = mock_api

            result = runner.invoke(cli_app, ["search", "test", "--gguf"], input="q")

            assert "model-gguf" in result.stdout


class TestHelperFunctions:
    """Tests for CLI helper functions."""

    def test_format_size_gb(self):
        """Size formatting in GB."""
        from hfl.cli.main import _format_size

        assert "5.0 GB" in _format_size(5 * 1024**3)
        assert "1.5 GB" in _format_size(int(1.5 * 1024**3))

    def test_format_size_mb(self):
        """Size formatting in MB."""
        from hfl.cli.main import _format_size

        assert "500 MB" in _format_size(500 * 1024**2)

    def test_format_size_zero(self):
        """Zero size formatting."""
        from hfl.cli.main import _format_size

        assert _format_size(0) == "N/A"

    def test_display_model_row(self, capsys):
        """Verifies model row format."""
        from hfl.cli.main import _display_model_row

        mock_model = MagicMock()
        mock_model.id = "org/model"
        mock_model.downloads = 1500000
        mock_model.likes = 100
        mock_model.siblings = [MagicMock(rfilename="model.gguf")]
        mock_model.pipeline_tag = "text-generation"

        _display_model_row(mock_model, 1)
        # No failure = test passes


class TestCLIApp:
    """General tests for the CLI application."""

    def test_help(self, runner, cli_app):
        """Verifies general help."""
        result = runner.invoke(cli_app, ["--help"])

        assert result.exit_code == 0
        assert "pull" in result.stdout
        assert "run" in result.stdout
        assert "serve" in result.stdout
        assert "list" in result.stdout
        assert "search" in result.stdout
        assert "rm" in result.stdout
        assert "inspect" in result.stdout

    def test_pull_help(self, runner):
        """Verifies pull help."""
        # Import fresh app to avoid interference from mock_llama_cpp
        from hfl.cli.main import app
        result = runner.invoke(app, ["pull", "--help"])

        assert result.exit_code == 0
        assert "--quantize" in result.stdout
        assert "--format" in result.stdout

    def test_search_help(self, runner):
        """Verifies search help."""
        from hfl.cli.main import app
        result = runner.invoke(app, ["search", "--help"])

        assert result.exit_code == 0
        assert "--limit" in result.stdout
        assert "--page-size" in result.stdout
        assert "--gguf" in result.stdout
        assert "--sort" in result.stdout

    def test_run_help(self, runner):
        """Verifies run help."""
        from hfl.cli.main import app
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "--backend" in result.stdout
        assert "--ctx" in result.stdout
        assert "--system" in result.stdout

    def test_serve_help(self, runner):
        """Verifies serve help."""
        from hfl.cli.main import app
        result = runner.invoke(app, ["serve", "--help"])

        assert result.exit_code == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--model" in result.stdout
