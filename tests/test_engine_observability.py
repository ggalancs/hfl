# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for engine observability module."""

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from hfl.engine.observability import (
    EngineObserver,
    InferenceMetrics,
    get_observer,
)


class TestInferenceMetrics:
    """Tests for InferenceMetrics class."""

    def test_initial_state(self):
        """Should initialize with correct defaults."""
        metrics = InferenceMetrics()
        assert metrics.start_time > 0
        assert metrics.end_time == 0.0
        assert metrics.tokens_prompt == 0
        assert metrics.tokens_generated == 0
        assert metrics.model_name == ""
        assert metrics.operation == "generate"

    def test_duration_seconds_incomplete(self):
        """Duration should calculate from start to now when incomplete."""
        metrics = InferenceMetrics()
        time.sleep(0.1)
        duration = metrics.duration_seconds
        assert duration >= 0.1
        assert duration < 0.5

    def test_duration_seconds_complete(self):
        """Duration should calculate from start to end when complete."""
        metrics = InferenceMetrics()
        metrics.start_time = 100.0
        metrics.end_time = 101.5
        assert metrics.duration_seconds == 1.5

    def test_duration_ms(self):
        """Should convert duration to milliseconds."""
        metrics = InferenceMetrics()
        metrics.start_time = 100.0
        metrics.end_time = 100.5
        assert metrics.duration_ms == 500.0

    def test_tokens_per_second(self):
        """Should calculate tokens per second."""
        metrics = InferenceMetrics()
        metrics.start_time = 100.0
        metrics.end_time = 102.0  # 2 seconds
        metrics.tokens_generated = 100  # 100 tokens
        assert metrics.tokens_per_second == 50.0

    def test_tokens_per_second_zero_duration(self):
        """Should return 0 for zero duration."""
        metrics = InferenceMetrics()
        metrics.start_time = 100.0
        metrics.end_time = 100.0
        metrics.tokens_generated = 100
        assert metrics.tokens_per_second == 0.0

    def test_tokens_per_second_zero_tokens(self):
        """Should return 0 for zero tokens."""
        metrics = InferenceMetrics()
        metrics.start_time = 100.0
        metrics.end_time = 102.0
        metrics.tokens_generated = 0
        assert metrics.tokens_per_second == 0.0

    def test_time_to_first_token_placeholder(self):
        """time_to_first_token_ms should return None (placeholder)."""
        metrics = InferenceMetrics()
        assert metrics.time_to_first_token_ms is None

    def test_complete_marks_end_time(self):
        """complete() should set end_time."""
        metrics = InferenceMetrics()
        metrics.complete()
        assert metrics.end_time > 0

    def test_complete_extracts_result_metrics(self):
        """complete() should extract metrics from result."""

        @dataclass
        class MockResult:
            tokens_prompt: int = 50
            tokens_generated: int = 100

        metrics = InferenceMetrics()
        metrics.complete(MockResult())
        assert metrics.tokens_prompt == 50
        assert metrics.tokens_generated == 100

    def test_to_dict(self):
        """to_dict() should return all metrics."""
        metrics = InferenceMetrics(
            model_name="test-model",
            operation="chat",
        )
        metrics.start_time = 100.0
        metrics.end_time = 101.0
        metrics.tokens_prompt = 10
        metrics.tokens_generated = 50

        result = metrics.to_dict()

        assert result["model"] == "test-model"
        assert result["operation"] == "chat"
        assert result["duration_ms"] == 1000.0
        assert result["tokens_prompt"] == 10
        assert result["tokens_generated"] == 50
        assert result["tokens_per_second"] == 50.0


class TestEngineObserver:
    """Tests for EngineObserver class."""

    def test_init_with_model_name(self):
        """Should initialize with model name."""
        observer = EngineObserver("my-model")
        assert observer.model_name == "my-model"
        assert observer._enabled is True

    def test_observe_inference_yields_metrics(self):
        """observe_inference should yield InferenceMetrics."""
        observer = EngineObserver("test")

        with observer.observe_inference("generate", emit_events=False) as metrics:
            assert isinstance(metrics, InferenceMetrics)
            assert metrics.model_name == "test"
            assert metrics.operation == "generate"

    def test_observe_inference_records_completion(self):
        """observe_inference should record completion metrics."""
        observer = EngineObserver("test")

        with observer.observe_inference(emit_events=False, record_metrics=False) as metrics:
            metrics.tokens_generated = 100

        # Metrics should be completed after context exit
        assert metrics.end_time > 0

    def test_observe_inference_handles_exception(self):
        """observe_inference should log exception and re-raise."""
        observer = EngineObserver("test")

        with pytest.raises(ValueError):
            with observer.observe_inference(emit_events=False):
                raise ValueError("Test error")

    @patch("hfl.engine.observability.logger")
    def test_observe_inference_logs_completion(self, mock_logger):
        """observe_inference should log on completion."""
        observer = EngineObserver("test")

        with observer.observe_inference(emit_events=False, record_metrics=False):
            pass

        mock_logger.info.assert_called()

    @patch("hfl.engine.observability.logger")
    def test_observe_inference_logs_error(self, mock_logger):
        """observe_inference should log on error."""
        observer = EngineObserver("test")

        with pytest.raises(RuntimeError):
            with observer.observe_inference(emit_events=False):
                raise RuntimeError("Boom")

        mock_logger.error.assert_called()

    def test_observe_model_load_yields_info(self):
        """observe_model_load should yield load info dict."""
        observer = EngineObserver("test-model")

        with observer.observe_model_load("/path/to/model") as load_info:
            assert "model_path" in load_info
            assert "model_name" in load_info
            assert "start_time" in load_info
            assert load_info["model_path"] == "/path/to/model"
            assert load_info["model_name"] == "test-model"

    def test_observe_model_load_records_duration(self):
        """observe_model_load should record duration on success."""
        observer = EngineObserver("test")

        with observer.observe_model_load("/path") as load_info:
            time.sleep(0.05)
            load_info["backend"] = "llama-cpp"

        assert "duration_ms" in load_info
        assert load_info["duration_ms"] >= 50
        assert load_info["backend"] == "llama-cpp"

    def test_observe_model_load_handles_exception(self):
        """observe_model_load should handle exception."""
        observer = EngineObserver("test")

        with pytest.raises(IOError):
            with observer.observe_model_load("/bad/path"):
                raise IOError("Cannot read model")

    @patch("hfl.engine.observability.logger")
    def test_observe_model_unload_logs(self, mock_logger):
        """observe_model_unload should log."""
        observer = EngineObserver("test-model")
        observer.observe_model_unload()

        mock_logger.info.assert_called()
        call_args = str(mock_logger.info.call_args)
        assert "test-model" in call_args

    @patch("hfl.events.emit")
    def test_emits_generation_started_event(self, mock_emit):
        """Should emit GENERATION_STARTED event."""
        observer = EngineObserver("test")

        with observer.observe_inference("chat", record_metrics=False):
            pass

        # Check that emit was called with GENERATION_STARTED
        calls = [str(c) for c in mock_emit.call_args_list]
        assert any("GENERATION_STARTED" in c for c in calls)

    @patch("hfl.events.emit")
    def test_emits_generation_completed_event(self, mock_emit):
        """Should emit GENERATION_COMPLETED event."""
        observer = EngineObserver("test")

        with observer.observe_inference("generate", record_metrics=False):
            pass

        calls = [str(c) for c in mock_emit.call_args_list]
        assert any("GENERATION_COMPLETED" in c for c in calls)

    @patch("hfl.events.emit")
    def test_emits_generation_failed_event(self, mock_emit):
        """Should emit GENERATION_FAILED event on error."""
        observer = EngineObserver("test")

        with pytest.raises(Exception):
            with observer.observe_inference(record_metrics=False):
                raise Exception("Test error")

        calls = [str(c) for c in mock_emit.call_args_list]
        assert any("GENERATION_FAILED" in c for c in calls)

    @patch("hfl.events.emit")
    def test_emits_model_loading_event(self, mock_emit):
        """Should emit MODEL_LOADING event."""
        observer = EngineObserver("test")

        with observer.observe_model_load("/path"):
            pass

        calls = [str(c) for c in mock_emit.call_args_list]
        assert any("MODEL_LOADING" in c for c in calls)

    @patch("hfl.events.emit")
    def test_emits_model_loaded_event(self, mock_emit):
        """Should emit MODEL_LOADED event."""
        observer = EngineObserver("test")

        with observer.observe_model_load("/path"):
            pass

        calls = [str(c) for c in mock_emit.call_args_list]
        assert any("MODEL_LOADED" in c for c in calls)

    @patch("hfl.events.emit")
    def test_emits_model_unloaded_event(self, mock_emit):
        """Should emit MODEL_UNLOADED event."""
        observer = EngineObserver("test")
        observer.observe_model_unload()

        mock_emit.assert_called()
        calls = [str(c) for c in mock_emit.call_args_list]
        assert any("MODEL_UNLOADED" in c for c in calls)

    @patch("hfl.metrics.get_metrics")
    def test_records_generation_metrics(self, mock_get_metrics):
        """Should record generation metrics."""
        mock_metrics = MagicMock()
        mock_get_metrics.return_value = mock_metrics

        observer = EngineObserver("test")

        with observer.observe_inference(emit_events=False):
            pass

        mock_metrics.record_generation.assert_called_once()

    @patch("hfl.metrics.get_metrics")
    def test_records_model_load_metrics(self, mock_get_metrics):
        """Should record model load metrics."""
        mock_metrics = MagicMock()
        mock_get_metrics.return_value = mock_metrics

        observer = EngineObserver("test-model")

        with observer.observe_model_load("/path"):
            pass

        mock_metrics.record_model_load.assert_called_once()

    @patch("hfl.metrics.get_metrics")
    def test_records_model_unload_metric(self, mock_get_metrics):
        """Should record model unload metric."""
        mock_metrics = MagicMock()
        mock_get_metrics.return_value = mock_metrics

        observer = EngineObserver("test")
        observer.observe_model_unload()

        mock_metrics.record_model_unload.assert_called_once()

    def test_disabled_observer_skips_events(self):
        """Disabled observer should skip event emission."""
        observer = EngineObserver("test")
        observer._enabled = False

        with patch("hfl.events.emit") as mock_emit:
            with observer.observe_inference():
                pass
            mock_emit.assert_not_called()


class TestGetObserver:
    """Tests for get_observer function."""

    def test_returns_observer(self):
        """get_observer should return EngineObserver."""
        observer = get_observer("my-model")
        assert isinstance(observer, EngineObserver)
        assert observer.model_name == "my-model"

    def test_default_model_name(self):
        """get_observer should use 'unknown' as default."""
        observer = get_observer()
        assert observer.model_name == "unknown"
