# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The `pooling` field on /api/embed, connected to something at last.

`engine/embedding_pooling.py` implemented mean / cls / last and said in
its own docstring that "HFL now accepts a ``pooling`` field on
``/api/embed``". The field did exist on `OllamaEmbedRequest`. The route
never passed it to the engine, and `TransformersEmbeddingEngine` had mean
pooling written into the tensor maths — so every request got mean
pooling, whatever it asked for.

That is the worst shape a half-finished feature takes: not an error, not
a missing endpoint, just a parameter that is accepted and discarded. A
caller serving a CLS-trained model would set `pooling="cls"`, receive
plausible unit vectors, and get quietly weaker retrieval with nothing to
look at.

So these tests assert the vectors actually differ per strategy. A test
that only checked "200 OK with pooling=cls" would have passed against the
broken version too.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hfl.engine.embedding_engine import LlamaCppEmbeddingEngine, TransformersEmbeddingEngine
from hfl.engine.embedding_pooling import POOLING_STRATEGIES, pool


class TestStrategiesProduceDifferentVectors:
    """Guarding the mechanism at its source.

    Three rows chosen so mean, first and last are all distinct — with
    identical rows every strategy agrees and the test proves nothing.
    """

    MATRIX = [[1.0, 0.0], [0.0, 1.0], [4.0, 4.0]]

    def test_cls_takes_the_first_token(self):
        assert pool(self.MATRIX, [1, 1, 1], "cls") == [1.0, 0.0]

    def test_last_takes_the_final_unmasked_token(self):
        assert pool(self.MATRIX, [1, 1, 1], "last") == [4.0, 4.0]

    def test_last_respects_padding(self):
        """The final *real* token, not the final row."""
        assert pool(self.MATRIX, [1, 1, 0], "last") == [0.0, 1.0]

    def test_mean_is_masked(self):
        assert pool(self.MATRIX, [1, 1, 0], "mean") == pytest.approx([0.5, 0.5])

    def test_the_three_disagree(self):
        vectors = [tuple(pool(self.MATRIX, [1, 1, 1], s)) for s in POOLING_STRATEGIES]
        assert len(set(vectors)) == 3, (
            "the strategies returned the same vector, so nothing downstream "
            "could tell them apart either"
        )


class TestTransformersEngineHonoursTheRequest:
    @staticmethod
    def _engine():
        torch = pytest.importorskip("torch", reason="transformers extra not installed")
        eng = TransformersEmbeddingEngine()
        eng._loaded = True
        eng._device = "cpu"
        eng._n_embd = 2

        # Two tokens, deliberately different, with no padding.
        hidden = torch.tensor([[[1.0, 0.0], [0.0, 3.0]]])
        outputs = MagicMock()
        outputs.last_hidden_state = hidden
        eng._model = MagicMock(return_value=outputs)

        encoded = {"attention_mask": torch.tensor([[1, 1]])}
        tok = MagicMock()
        tok.return_value = MagicMock(to=MagicMock(return_value=encoded))
        eng._tokenizer = tok
        return eng

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [("cls", [1.0, 0.0]), ("last", [0.0, 1.0]), ("mean", [0.316, 0.949])],
    )
    def test_each_strategy_reaches_the_maths(self, strategy, expected):
        """Vectors are L2-normalised, so compare against normalised targets."""
        engine = self._engine()
        result = engine.embed(["x"], pooling=strategy)
        assert result.embeddings[0] == pytest.approx(expected, abs=0.01)

    def test_the_default_is_still_mean(self):
        engine = self._engine()
        assert engine.embed(["x"]).embeddings[0] == pytest.approx(
            engine.embed(["x"], pooling="mean").embeddings[0]
        )

    def test_an_unknown_strategy_raises_instead_of_falling_back(self):
        """`pool()` silently falls back to mean on a bad name. At the engine
        boundary that would be the original bug wearing a new coat."""
        engine = self._engine()
        with pytest.raises(ValueError, match="unknown pooling"):
            engine.embed(["x"], pooling="bogus")


class TestLlamaCppRefusesRatherThanPretending:
    """llama.cpp pools inside the C library; the token matrix never reaches
    Python. Accepting `cls` and returning mean-pooled vectors labelled as
    CLS is exactly the failure this whole change is about."""

    @staticmethod
    def _engine():
        eng = LlamaCppEmbeddingEngine()
        eng._loaded = True
        eng._llm = MagicMock()
        eng._llm.embed = MagicMock(return_value=[0.6, 0.8])
        eng._n_embd = 2
        return eng

    @pytest.mark.parametrize("strategy", ["cls", "last"])
    def test_non_mean_is_refused(self, strategy):
        with pytest.raises(ValueError) as caught:
            self._engine().embed(["x"], pooling=strategy)
        message = str(caught.value)
        assert "llama.cpp" in message
        assert "transformers" in message, "the refusal must name the way forward"

    def test_mean_still_works(self):
        assert self._engine().embed(["x"], pooling="mean").embeddings[0] == [0.6, 0.8]


class TestTheRouteActuallyPassesItOn:
    """The link that was missing."""

    def test_the_request_model_validates_the_value(self):
        from pydantic import ValidationError

        from hfl.api.routes_embed import OllamaEmbedRequest

        assert OllamaEmbedRequest(model="m", input="x").pooling == "mean"
        assert OllamaEmbedRequest(model="m", input="x", pooling="cls").pooling == "cls"
        with pytest.raises(ValidationError):
            OllamaEmbedRequest(model="m", input="x", pooling="bogus")

    def test_the_handler_forwards_pooling_to_the_engine(self):
        """Read from the source, because the regression is an omission.

        The field existed and the call did not use it; nothing about the
        types would have caught that.
        """
        import inspect

        from hfl.api import routes_embed

        source = inspect.getsource(routes_embed.ollama_embed)
        assert "pooling=req.pooling" in source, (
            "the route builds its engine.embed call without the pooling the "
            "caller asked for — the parameter is accepted and discarded"
        )
