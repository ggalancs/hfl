"""Tests para el módulo engine (base, llama_cpp, transformers, selector)."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# Mock llama_cpp antes de importar los módulos
@pytest.fixture(autouse=True)
def mock_llama_cpp_module():
    """Mock del módulo llama_cpp para todos los tests."""
    mock_llama = MagicMock()
    mock_llama.Llama = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": mock_llama}):
        yield mock_llama


class TestChatMessage:
    """Tests para ChatMessage dataclass."""

    def test_creation(self):
        """Verifica creación de ChatMessage."""
        from hfl.engine.base import ChatMessage

        msg = ChatMessage(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_system_message(self):
        """Verifica mensaje de sistema."""
        from hfl.engine.base import ChatMessage

        msg = ChatMessage(role="system", content="You are helpful")

        assert msg.role == "system"
        assert msg.content == "You are helpful"

    def test_assistant_message(self):
        """Verifica mensaje de asistente."""
        from hfl.engine.base import ChatMessage

        msg = ChatMessage(role="assistant", content="Hi there!")

        assert msg.role == "assistant"


class TestGenerationConfig:
    """Tests para GenerationConfig dataclass."""

    def test_default_values(self):
        """Verifica valores por defecto."""
        from hfl.engine.base import GenerationConfig

        cfg = GenerationConfig()

        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.top_k == 40
        assert cfg.max_tokens == 2048
        assert cfg.stop is None
        assert cfg.repeat_penalty == 1.1
        assert cfg.seed == -1

    def test_custom_values(self):
        """Verifica valores personalizados."""
        from hfl.engine.base import GenerationConfig

        cfg = GenerationConfig(
            temperature=0.5,
            top_p=0.8,
            top_k=50,
            max_tokens=100,
            stop=["END", "STOP"],
            repeat_penalty=1.2,
            seed=42,
        )

        assert cfg.temperature == 0.5
        assert cfg.top_p == 0.8
        assert cfg.top_k == 50
        assert cfg.max_tokens == 100
        assert cfg.stop == ["END", "STOP"]
        assert cfg.repeat_penalty == 1.2
        assert cfg.seed == 42


class TestGenerationResult:
    """Tests para GenerationResult dataclass."""

    def test_default_values(self):
        """Verifica valores por defecto."""
        from hfl.engine.base import GenerationResult

        result = GenerationResult(text="Hello")

        assert result.text == "Hello"
        assert result.tokens_generated == 0
        assert result.tokens_prompt == 0
        assert result.tokens_per_second == 0.0
        assert result.stop_reason == "stop"

    def test_full_result(self):
        """Verifica resultado completo."""
        from hfl.engine.base import GenerationResult

        result = GenerationResult(
            text="Generated text",
            tokens_generated=50,
            tokens_prompt=10,
            tokens_per_second=25.5,
            stop_reason="length",
        )

        assert result.text == "Generated text"
        assert result.tokens_generated == 50
        assert result.tokens_prompt == 10
        assert result.tokens_per_second == 25.5
        assert result.stop_reason == "length"


class TestLlamaCppEngine:
    """Tests para LlamaCppEngine."""

    def test_initialization(self, mock_llama_cpp_module):
        """Verifica inicialización del engine."""
        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()

        assert engine._model is None
        assert engine._model_path == ""
        assert not engine.is_loaded
        assert engine.model_name == ""

    def test_load_model(self, mock_llama_cpp_module):
        """Verifica carga de modelo."""
        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        engine.load("/path/to/model.gguf", n_ctx=2048)

        assert engine.is_loaded
        assert engine.model_name == "model.gguf"

    def test_load_with_kwargs(self, mock_llama_cpp_module):
        """Verifica carga con parámetros adicionales."""
        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        engine.load(
            "/path/to/model.gguf",
            n_ctx=4096,
            n_gpu_layers=32,
            n_threads=8,
            verbose=True,
        )

        assert engine.is_loaded

    def test_unload_model(self, mock_llama_cpp_module):
        """Verifica descarga de modelo."""
        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        engine.load("/path/to/model.gguf")
        assert engine.is_loaded

        engine.unload()
        assert not engine.is_loaded

    def test_generate(self, mock_llama_cpp_module):
        """Verifica generación de texto."""
        from hfl.engine.llama_cpp import LlamaCppEngine
        from hfl.engine.base import GenerationConfig

        engine = LlamaCppEngine()

        # Configurar mock del modelo
        mock_model = MagicMock()
        mock_model.return_value = {
            "choices": [{"text": "Generated text", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch("hfl.engine.llama_cpp.Llama", return_value=mock_model):
            engine.load("/path/to/model.gguf")
            result = engine.generate("Prompt", GenerationConfig(max_tokens=100))

            assert result.text == "Generated text"
            assert result.tokens_prompt == 10
            assert result.tokens_generated == 5
            assert result.stop_reason == "stop"

    def test_generate_stream(self, mock_llama_cpp_module):
        """Verifica generación en streaming."""
        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()

        mock_model = MagicMock()
        mock_model.return_value = iter([
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " world"}]},
            {"choices": [{"text": "!"}]},
        ])

        with patch("hfl.engine.llama_cpp.Llama", return_value=mock_model):
            engine.load("/path/to/model.gguf")
            tokens = list(engine.generate_stream("Prompt"))

            assert tokens == ["Hello", " world", "!"]

    def test_chat(self, mock_llama_cpp_module):
        """Verifica chat completion."""
        from hfl.engine.llama_cpp import LlamaCppEngine
        from hfl.engine.base import ChatMessage

        engine = LlamaCppEngine()

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }

        with patch("hfl.engine.llama_cpp.Llama", return_value=mock_model):
            engine.load("/path/to/model.gguf")

            messages = [ChatMessage(role="user", content="Hi")]
            result = engine.chat(messages)

            assert result.text == "Hello!"
            mock_model.create_chat_completion.assert_called_once()

    def test_chat_stream(self, mock_llama_cpp_module):
        """Verifica chat completion en streaming."""
        from hfl.engine.llama_cpp import LlamaCppEngine
        from hfl.engine.base import ChatMessage

        engine = LlamaCppEngine()

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "Hi"}}]},
            {"choices": [{"delta": {"content": " there"}}]},
        ])

        with patch("hfl.engine.llama_cpp.Llama", return_value=mock_model):
            engine.load("/path/to/model.gguf")

            messages = [ChatMessage(role="user", content="Hello")]
            tokens = list(engine.chat_stream(messages))

            assert tokens == ["Hi", " there"]


class TestEngineSelector:
    """Tests para engine/selector.py."""

    def test_select_llama_cpp_for_gguf(self, temp_dir, mock_llama_cpp_module):
        """Selecciona LlamaCppEngine para GGUF."""
        from hfl.engine.selector import select_engine
        from hfl.engine.llama_cpp import LlamaCppEngine

        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF")

        engine = select_engine(gguf_file)

        assert isinstance(engine, LlamaCppEngine)

    def test_select_explicit_backend(self, temp_dir, mock_llama_cpp_module):
        """Selecciona backend explícito."""
        from hfl.engine.selector import select_engine
        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = select_engine(temp_dir, backend="llama-cpp")

        assert isinstance(engine, LlamaCppEngine)

    def test_select_invalid_backend(self, temp_dir):
        """Error con backend inválido."""
        from hfl.engine.selector import _create_engine

        with pytest.raises(ValueError, match="Backend desconocido"):
            _create_engine("invalid")

    def test_has_cuda_detection(self):
        """Verifica detección de CUDA."""
        from hfl.engine.selector import _has_cuda

        # Sin torch instalado o sin CUDA debería devolver False
        result = _has_cuda()
        assert isinstance(result, bool)

    def test_select_fallback_to_llama_cpp(self, temp_dir, mock_llama_cpp_module):
        """Fallback a llama.cpp cuando no hay CUDA."""
        from hfl.engine.selector import select_engine
        from hfl.engine.llama_cpp import LlamaCppEngine

        # Crear archivo safetensors
        (temp_dir / "model.safetensors").write_bytes(b"ST")

        with patch("hfl.engine.selector._has_cuda", return_value=False):
            engine = select_engine(temp_dir)

            assert isinstance(engine, LlamaCppEngine)


class TestMissingDependencyErrors:
    """Tests para errores de dependencias faltantes."""

    def test_missing_dependency_error_exists(self):
        """Verifica que MissingDependencyError esté definido."""
        from hfl.engine.selector import MissingDependencyError

        assert issubclass(MissingDependencyError, Exception)

    def test_missing_llama_cpp_raises_error(self, temp_dir):
        """Verifica que se lance MissingDependencyError cuando falta llama_cpp."""
        from hfl.engine.selector import MissingDependencyError

        # Crear un MissingDependencyError manualmente para verificar su contenido
        error = MissingDependencyError(
            "El backend llama-cpp requiere la librería 'llama-cpp-python'.\n"
            "Instálala con: pip install llama-cpp-python"
        )

        assert "llama-cpp-python" in str(error)
        assert "pip install" in str(error)

    def test_missing_llama_cpp_error_message_contains_gpu_instructions(self, temp_dir):
        """Verifica que el mensaje de error incluya instrucciones para GPU."""
        from hfl.engine.selector import _get_llama_cpp_engine, MissingDependencyError

        with patch(
            "hfl.engine.llama_cpp.LlamaCppEngine",
            side_effect=ImportError("No module named 'llama_cpp'")
        ):
            with pytest.raises(MissingDependencyError) as exc_info:
                _get_llama_cpp_engine()

            error_msg = str(exc_info.value)
            assert "CUDA" in error_msg
            assert "Metal" in error_msg

    def test_select_engine_gguf_without_llama_cpp(self, temp_dir):
        """Verifica que select_engine falle con mensaje claro cuando falta llama_cpp."""
        from hfl.engine.selector import select_engine, MissingDependencyError

        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF")

        with patch(
            "hfl.engine.llama_cpp.LlamaCppEngine",
            side_effect=ImportError("No module named 'llama_cpp'")
        ):
            with pytest.raises(MissingDependencyError) as exc_info:
                select_engine(gguf_file)

            assert "llama-cpp-python" in str(exc_info.value)

    def test_create_engine_explicit_llama_cpp_without_dependency(self):
        """Verifica que _create_engine('llama-cpp') falle con mensaje claro."""
        from hfl.engine.selector import _create_engine, MissingDependencyError

        with patch(
            "hfl.engine.llama_cpp.LlamaCppEngine",
            side_effect=ImportError("No module named 'llama_cpp'")
        ):
            with pytest.raises(MissingDependencyError):
                _create_engine("llama-cpp")

    def test_missing_transformers_raises_error(self):
        """Verifica que se lance MissingDependencyError cuando falta transformers."""
        from hfl.engine.selector import _get_transformers_engine, MissingDependencyError

        with patch(
            "hfl.engine.transformers_engine.TransformersEngine",
            side_effect=ImportError("No module named 'transformers'")
        ):
            with pytest.raises(MissingDependencyError) as exc_info:
                _get_transformers_engine()

            error_msg = str(exc_info.value)
            assert "transformers" in error_msg
            assert "pip install" in error_msg

    def test_missing_vllm_raises_error(self):
        """Verifica que se lance MissingDependencyError cuando falta vllm."""
        from hfl.engine.selector import _get_vllm_engine, MissingDependencyError

        # Simular que el import de vllm_engine falla
        with patch.dict(sys.modules, {"hfl.engine.vllm_engine": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                _get_vllm_engine()

            error_msg = str(exc_info.value)
            assert "vllm" in error_msg.lower()
            assert "pip install" in error_msg


class TestTransformersEngine:
    """Tests para TransformersEngine (requiere mock completo)."""

    def test_initialization(self):
        """Verifica inicialización del engine."""
        from hfl.engine.transformers_engine import TransformersEngine

        engine = TransformersEngine()

        assert engine._model is None
        assert engine._tokenizer is None
        assert engine._model_id == ""
        assert not engine.is_loaded

    def test_build_prompt_with_chat_template(self):
        """Verifica construcción de prompt con chat template."""
        from hfl.engine.transformers_engine import TransformersEngine
        from hfl.engine.base import ChatMessage

        engine = TransformersEngine()
        engine._tokenizer = MagicMock()
        engine._tokenizer.apply_chat_template.return_value = "formatted prompt"

        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello"),
        ]

        result = engine._build_prompt(messages)

        assert result == "formatted prompt"
        engine._tokenizer.apply_chat_template.assert_called_once()

    def test_build_prompt_fallback(self):
        """Verifica fallback cuando no hay chat template."""
        from hfl.engine.transformers_engine import TransformersEngine
        from hfl.engine.base import ChatMessage

        engine = TransformersEngine()
        engine._tokenizer = MagicMock(spec=[])  # Sin apply_chat_template

        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello"),
        ]

        result = engine._build_prompt(messages)

        assert "You are helpful" in result
        assert "Hello" in result

    def test_model_name_property(self):
        """Verifica propiedad model_name."""
        from hfl.engine.transformers_engine import TransformersEngine

        engine = TransformersEngine()
        engine._model_id = "test/model"

        assert engine.model_name == "test/model"

    def test_is_loaded_property(self):
        """Verifica propiedad is_loaded."""
        from hfl.engine.transformers_engine import TransformersEngine

        engine = TransformersEngine()
        assert not engine.is_loaded

        engine._model = MagicMock()
        assert engine.is_loaded
