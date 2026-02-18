"""Tests para el módulo CLI (main commands)."""

import pytest
import sys
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def mock_llama_cpp():
    """Mock llama_cpp para todos los tests."""
    mock = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": mock}):
        yield mock


@pytest.fixture
def runner():
    """Runner para tests de CLI."""
    return CliRunner()


@pytest.fixture
def cli_app():
    """Aplicación CLI para tests."""
    from hfl.cli.main import app
    return app


class TestVersionCommand:
    """Tests para comando version."""

    def test_version(self, runner, cli_app):
        """Verifica comando version."""
        result = runner.invoke(cli_app, ["version"])

        assert result.exit_code == 0
        assert "hfl" in result.stdout
        assert "0.1.0" in result.stdout
        assert "HRUL" in result.stdout  # License reference


class TestListCommand:
    """Tests para comando list."""

    def test_list_empty(self, runner, cli_app, temp_config):
        """Lista vacía de modelos."""
        result = runner.invoke(cli_app, ["list"])

        assert result.exit_code == 0
        assert "No hay modelos descargados" in result.stdout

    def test_list_with_models(self, runner, cli_app, populated_registry):
        """Lista con modelos."""
        result = runner.invoke(cli_app, ["list"])

        assert result.exit_code == 0
        # La tabla de Rich trunca nombres largos con "…" (U+2026)
        # Verificamos prefijos o la cuantización que siempre aparece
        assert "Q4_K_M" in result.stdout
        assert "Q5_K_M" in result.stdout
        assert "test-org" in result.stdout or "test-model" in result.stdout
        assert "other-org" in result.stdout or "another" in result.stdout


class TestInspectCommand:
    """Tests para comando inspect."""

    def test_inspect_not_found(self, runner, cli_app, temp_config):
        """Inspeccionar modelo inexistente."""
        result = runner.invoke(cli_app, ["inspect", "nonexistent"])

        assert result.exit_code == 1
        assert "Modelo no encontrado" in result.stdout

    def test_inspect_success(self, runner, cli_app, populated_registry):
        """Inspeccionar modelo existente."""
        result = runner.invoke(cli_app, ["inspect", "test-model-q4_k_m"])

        assert result.exit_code == 0
        assert "test-model-q4_k_m" in result.stdout
        assert "test-org/test-model" in result.stdout
        assert "gguf" in result.stdout.lower()


class TestAliasCommand:
    """Tests para comando alias."""

    def test_alias_model_not_found(self, runner, cli_app, temp_config):
        """Alias a modelo inexistente."""
        result = runner.invoke(cli_app, ["alias", "nonexistent", "myalias"])

        assert result.exit_code == 1
        assert "Modelo no encontrado" in result.stdout

    def test_alias_success(self, runner, cli_app, temp_config, temp_dir):
        """Asignar alias exitosamente."""
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
        assert "Alias asignado" in result.stdout
        assert "short" in result.stdout

        # Verificar que el alias funciona
        registry = ModelRegistry()
        model = registry.get("short")
        assert model is not None
        assert model.name == "very-long-model-name-q4_k_m"

    def test_alias_duplicate_rejected(self, runner, cli_app, temp_config, temp_dir):
        """Alias duplicado es rechazado."""
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
        assert "ya está en uso" in result.stdout

    def test_run_by_alias(self, runner, cli_app, temp_config, temp_dir):
        """Ejecutar modelo por alias."""
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

            assert "Cargando" in result.stdout

    def test_inspect_by_alias(self, runner, cli_app, temp_config, temp_dir):
        """Inspeccionar modelo por alias."""
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
    """Tests para comando rm."""

    def test_rm_not_found(self, runner, cli_app, temp_config):
        """Eliminar modelo inexistente."""
        result = runner.invoke(cli_app, ["rm", "nonexistent"])

        assert result.exit_code == 1
        assert "Modelo no encontrado" in result.stdout

    def test_rm_cancelled(self, runner, cli_app, populated_registry):
        """Cancelar eliminación."""
        result = runner.invoke(cli_app, ["rm", "test-model-q4_k_m"], input="n\n")

        assert result.exit_code == 0
        # El modelo debería seguir existiendo
        from hfl.models.registry import ModelRegistry
        registry = ModelRegistry()
        assert registry.get("test-model-q4_k_m") is not None

    def test_rm_confirmed(self, runner, cli_app, temp_config, temp_dir):
        """Confirmar eliminación."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        # Crear archivo temporal para el modelo
        model_path = temp_dir / "test-model"
        model_path.mkdir()
        (model_path / "model.gguf").write_bytes(b"GGUF")

        # Registrar modelo con path real
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
        assert "Eliminado" in result.stdout

        # Verificar que se eliminó del registro
        registry = ModelRegistry()
        assert registry.get("rm-test-model") is None


class TestPullCommand:
    """Tests para comando pull."""

    def test_pull_success(self, runner, cli_app, temp_config):
        """Pull exitoso de modelo GGUF."""
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

                # Usar --skip-license para evitar llamadas a HuggingFace API
                result = runner.invoke(cli_app, ["pull", "test/model", "--skip-license"])

                assert result.exit_code == 0
                assert "Resolviendo" in result.stdout

    def test_pull_with_quantize_option(self, runner, cli_app, temp_config):
        """Pull con opción de cuantización."""
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

                # Usar --skip-license para evitar llamadas a HuggingFace API
                result = runner.invoke(cli_app, ["pull", "test/model", "-q", "Q5_K_M", "--skip-license"])

                mock_resolve.assert_called_once()


class TestRunCommand:
    """Tests para comando run."""

    def test_run_model_not_found(self, runner, cli_app, temp_config):
        """Run con modelo inexistente."""
        result = runner.invoke(cli_app, ["run", "nonexistent"])

        assert result.exit_code == 1
        assert "Modelo no encontrado" in result.stdout

    def test_run_missing_dependency_shows_helpful_error(self, runner, cli_app, temp_config, temp_dir):
        """Run muestra error útil cuando falta dependencia."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest
        from hfl.engine.selector import MissingDependencyError

        # Crear archivo de modelo
        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        # Registrar modelo
        registry = ModelRegistry()
        registry.add(ModelManifest(
            name="missing-dep-test",
            repo_id="test-org/model",
            local_path=str(model_path),
            format="gguf",
            size_bytes=100,
            quantization="Q4_K_M",
        ))

        # Simular que falta llama_cpp (parchear en selector donde se importa)
        with patch("hfl.engine.selector.select_engine") as mock_select:
            mock_select.side_effect = MissingDependencyError(
                "El backend llama-cpp requiere 'llama-cpp-python'.\n"
                "Instálala con: pip install llama-cpp-python"
            )

            result = runner.invoke(cli_app, ["run", "missing-dep-test"])

            assert result.exit_code == 1
            assert "Dependencia faltante" in result.stdout
            assert "llama-cpp-python" in result.stdout
            assert "pip install" in result.stdout

    def test_run_with_exit(self, runner, cli_app, temp_config, temp_dir):
        """Run con salida inmediata."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        # Crear archivo de modelo
        model_path = temp_dir / "test-model.gguf"
        model_path.write_bytes(b"GGUF")

        # Registrar modelo
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

            assert "Cargando" in result.stdout

    def test_run_streaming_with_special_characters(self, runner, cli_app, temp_config, temp_dir):
        """Run con caracteres especiales en la respuesta (evita errores de Rich markup)."""
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
            # Respuesta con caracteres que podrían causar errores de Rich markup
            mock_engine.chat_stream.return_value = iter([
                "Here's code: ",
                "[bold]",  # Podría interpretarse como markup de Rich
                " and ",
                "[/]",     # Podría causar MarkupError
                " brackets",
            ])
            mock_select.return_value = mock_engine

            result = runner.invoke(cli_app, ["run", "special-char-model"], input="test\n/exit\n")

            # No debe haber errores de markup
            assert result.exit_code == 0
            assert "MarkupError" not in result.stdout
            assert "Traceback" not in result.stdout

    def test_run_streaming_complete_response(self, runner, cli_app, temp_config, temp_dir):
        """Run captura respuesta completa del streaming."""
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
        """Run maneja respuesta vacía correctamente."""
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
            mock_engine.chat_stream.return_value = iter([])  # Respuesta vacía
            mock_select.return_value = mock_engine

            result = runner.invoke(cli_app, ["run", "empty-response-model"], input="Hi\n/exit\n")

            assert result.exit_code == 0
            assert "Traceback" not in result.stdout

    def test_run_multi_turn_conversation(self, runner, cli_app, temp_config, temp_dir):
        """Run maneja conversación multi-turno."""
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
            # Simular múltiples respuestas
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
        """Run reconoce varios comandos de salida."""
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

                assert result.exit_code == 0, f"Fallo con comando {exit_cmd}"
                assert "Sesión terminada" in result.stdout

    def test_run_empty_input_ignored(self, runner, cli_app, temp_config, temp_dir):
        """Run ignora entradas vacías."""
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

            # Enviar líneas vacías que deberían ignorarse
            result = runner.invoke(
                cli_app,
                ["run", "empty-input-model"],
                input="\n\n\nHello\n/exit\n"
            )

            assert result.exit_code == 0
            # chat_stream solo debe llamarse una vez (para "Hello")
            assert mock_engine.chat_stream.call_count == 1

    def test_run_with_system_prompt(self, runner, cli_app, temp_config, temp_dir):
        """Run con prompt de sistema."""
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
            # Verificar que se llamó con un mensaje de sistema
            call_args = mock_engine.chat_stream.call_args
            messages = call_args[0][0]
            assert any(m.role == "system" for m in messages)

    def test_run_unicode_response(self, runner, cli_app, temp_config, temp_dir):
        """Run maneja respuestas con Unicode correctamente."""
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
            # Respuesta con emojis y caracteres especiales
            mock_engine.chat_stream.return_value = iter([
                "¡Hola! ", "こんにちは ", "🎉", " Привет"
            ])
            mock_select.return_value = mock_engine

            result = runner.invoke(cli_app, ["run", "unicode-model"], input="Hi\n/exit\n")

            assert result.exit_code == 0
            assert "Traceback" not in result.stdout


class TestServeCommand:
    """Tests para comando serve."""

    def test_serve_imports(self, cli_app):
        """Verifica que serve importa correctamente."""
        from hfl.cli.main import serve
        assert callable(serve)


class TestSearchCommand:
    """Tests para comando search."""

    def test_search_too_short(self, runner, cli_app, temp_config):
        """Búsqueda con texto muy corto."""
        result = runner.invoke(cli_app, ["search", "ab"])

        assert result.exit_code == 1
        assert "al menos 3 caracteres" in result.stdout

    def test_search_no_results(self, runner, cli_app, temp_config):
        """Búsqueda sin resultados."""
        with patch("huggingface_hub.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.list_models.return_value = []
            mock_api_class.return_value = mock_api

            result = runner.invoke(cli_app, ["search", "xyznonexistent"])

            assert result.exit_code == 0
            assert "No se encontraron modelos" in result.stdout

    def test_search_with_results(self, runner, cli_app, temp_config):
        """Búsqueda con resultados."""
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
        """Búsqueda solo GGUF."""
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
    """Tests para funciones auxiliares del CLI."""

    def test_format_size_gb(self):
        """Formateo de tamaño en GB."""
        from hfl.cli.main import _format_size

        assert "5.0 GB" in _format_size(5 * 1024**3)
        assert "1.5 GB" in _format_size(int(1.5 * 1024**3))

    def test_format_size_mb(self):
        """Formateo de tamaño en MB."""
        from hfl.cli.main import _format_size

        assert "500 MB" in _format_size(500 * 1024**2)

    def test_format_size_zero(self):
        """Formateo de tamaño cero."""
        from hfl.cli.main import _format_size

        assert _format_size(0) == "N/A"

    def test_display_model_row(self, capsys):
        """Verifica formato de fila de modelo."""
        from hfl.cli.main import _display_model_row

        mock_model = MagicMock()
        mock_model.id = "org/model"
        mock_model.downloads = 1500000
        mock_model.likes = 100
        mock_model.siblings = [MagicMock(rfilename="model.gguf")]
        mock_model.pipeline_tag = "text-generation"

        _display_model_row(mock_model, 1)
        # No falla = test pasa


class TestCLIApp:
    """Tests generales para la aplicación CLI."""

    def test_help(self, runner, cli_app):
        """Verifica ayuda general."""
        result = runner.invoke(cli_app, ["--help"])

        assert result.exit_code == 0
        assert "pull" in result.stdout
        assert "run" in result.stdout
        assert "serve" in result.stdout
        assert "list" in result.stdout
        assert "search" in result.stdout
        assert "rm" in result.stdout
        assert "inspect" in result.stdout

    def test_pull_help(self, runner, cli_app):
        """Verifica ayuda de pull."""
        result = runner.invoke(cli_app, ["pull", "--help"])

        assert result.exit_code == 0
        assert "--quantize" in result.stdout
        assert "--format" in result.stdout

    def test_search_help(self, runner, cli_app):
        """Verifica ayuda de search."""
        result = runner.invoke(cli_app, ["search", "--help"])

        assert result.exit_code == 0
        assert "--limit" in result.stdout
        assert "--page-size" in result.stdout
        assert "--gguf" in result.stdout
        assert "--sort" in result.stdout

    def test_run_help(self, runner, cli_app):
        """Verifica ayuda de run."""
        result = runner.invoke(cli_app, ["run", "--help"])

        assert result.exit_code == 0
        assert "--backend" in result.stdout
        assert "--ctx" in result.stdout
        assert "--system" in result.stdout

    def test_serve_help(self, runner, cli_app):
        """Verifica ayuda de serve."""
        result = runner.invoke(cli_app, ["serve", "--help"])

        assert result.exit_code == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--model" in result.stdout
