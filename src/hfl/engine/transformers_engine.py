# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Backend based on HuggingFace Transformers.

Uses the model in its native format (safetensors) with GPU.
Supports dynamic quantization via bitsandbytes.
"""

import logging
import time
from typing import Iterator

from hfl.engine.base import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
)

logger = logging.getLogger(__name__)


class TransformersEngine(InferenceEngine):
    """HuggingFace Transformers inference engine."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_id = ""

    def load(self, model_path: str, **kwargs) -> None:
        """
        Loads a model from local directory or HF repo.

        Args:
            model_path: Local path or HuggingFace repo_id
            **kwargs:
                quantization: "4bit", "8bit", None
                device_map: "auto" (default), "cpu", "cuda:0"
                torch_dtype: "auto", "float16", "bfloat16"
                max_memory: dict of maximum memory per device
                trust_remote_code: bool (default False)
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from hfl.security import remote_code_allowed

        quant = kwargs.get("quantization")
        # trust_remote_code executes Python shipped in the model repo. Honour
        # the request only when the operator opted in via HFL_ALLOW_REMOTE_CODE,
        # so an untrusted caller can never turn model loading into RCE.
        load_kwargs = {
            "device_map": kwargs.get("device_map", "auto"),
            "torch_dtype": kwargs.get("torch_dtype", "auto"),
            "trust_remote_code": bool(kwargs.get("trust_remote_code", False))
            and remote_code_allowed(),
        }

        if quant == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quant == "8bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load with partial failure cleanup to prevent resource leaks
        logger.info("Loading transformers model: %s", model_path)
        logger.debug("Load config: quant=%s, device_map=%s", quant, load_kwargs.get("device_map"))

        start_time = time.perf_counter()
        tokenizer = None
        model = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            # Only assign to instance after both succeed
            self._tokenizer = tokenizer
            self._model = model
            self._model_id = model_path
            elapsed = time.perf_counter() - start_time
            logger.info("Model loaded in %.2fs: %s", elapsed, model_path)
        except Exception as e:
            logger.error("Failed to load model %s: %s", model_path, e)
            # Cleanup partial state on failure
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def unload(self) -> None:
        if self._model:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            # Force garbage collection and GPU memory cleanup
            import gc

            gc.collect()

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass  # Ignore if torch not available or CUDA errors

    def _build_prompt(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> str:
        """Builds the prompt using the tokenizer's chat template.

        When ``tools`` is provided, they are forwarded to the tokenizer's
        ``apply_chat_template`` so models that have tool-aware templates
        (qwen, llama3, mistral, ...) emit their native tool-call markers.
        """
        msgs: list[dict] = []
        for m in messages:
            entry: dict = {"role": m.role, "content": m.content or ""}
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            if m.name:
                entry["name"] = m.name
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            msgs.append(entry)

        if hasattr(self._tokenizer, "apply_chat_template"):
            template_kwargs: dict = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools:
                template_kwargs["tools"] = tools
            try:
                return self._tokenizer.apply_chat_template(msgs, **template_kwargs)
            except TypeError:
                # Older transformers / templates without ``tools`` kwarg
                template_kwargs.pop("tools", None)
                return self._tokenizer.apply_chat_template(msgs, **template_kwargs)

        # Generic fallback (no template, no tool awareness)
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"<<SYS>>{m.content}<</SYS>>")
            elif m.role == "user":
                parts.append(f"[INST] {m.content} [/INST]")
            elif m.role == "assistant":
                parts.append(m.content or "")
            elif m.role == "tool":
                parts.append(f"[TOOL {m.name}] {m.content or ''}")
        return "\n".join(parts)

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        import torch

        cfg = config or GenerationConfig()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=cfg.max_tokens,
                temperature=cfg.temperature if cfg.temperature > 0 else None,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repetition_penalty=cfg.repeat_penalty,
                do_sample=cfg.temperature > 0,
            )
        elapsed = time.perf_counter() - t0

        # Decode only the new tokens
        new_tokens = outputs[0][prompt_tokens:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        n_gen = len(new_tokens)

        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=prompt_tokens,
            tokens_per_second=n_gen / elapsed if elapsed > 0 else 0,
        )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        from threading import Event, Thread

        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        cfg = config or GenerationConfig()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": cfg.max_tokens,
            "temperature": cfg.temperature if cfg.temperature > 0 else None,
            "top_p": cfg.top_p,
            # ENG-6: match the non-streaming generate() so streaming samples
            # from the same distribution (top_k/repetition_penalty were
            # silently falling back to HF defaults on the streaming path).
            "top_k": cfg.top_k,
            "repetition_penalty": cfg.repeat_penalty,
            "do_sample": cfg.temperature > 0,
            "streamer": streamer,
        }

        # CON-3: cooperative cancellation. If the consumer closes this
        # generator early (client disconnect / stream timeout), the ``finally``
        # sets ``cancel`` and the criteria halts generation at the next token,
        # so the worker thread does not outlive the request. Guarded by
        # ``isinstance(..., type)`` so the unit tests' mocked ``transformers``
        # (a MagicMock, which cannot be subclassed) takes the no-op path.
        cancel = Event()
        if isinstance(StoppingCriteria, type):

            class _Cancelled(StoppingCriteria):  # type: ignore[misc, valid-type]
                def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
                    return cancel.is_set()

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_Cancelled()])

        # ENG-8: a bare Thread never propagates the worker's exception, so an
        # OOM / shape / CUDA failure mid-generation used to be swallowed and
        # the consumer hung forever on the streamer. Capture it and re-raise
        # after the loop; push the stop sentinel so the loop always terminates.
        worker_error: list[BaseException] = []

        def _run() -> None:
            try:
                self._model.generate(**gen_kwargs)
            except BaseException as exc:  # noqa: BLE001 - surfaced after the loop
                worker_error.append(exc)
                try:
                    streamer.end()
                except Exception:  # pragma: no cover - defensive
                    pass

        thread = Thread(target=_run)
        thread.start()

        clean_exit = False
        try:
            for text in streamer:
                if text:
                    yield text
            clean_exit = True
        finally:
            # On early close (GeneratorExit) or normal exhaustion, stop the
            # worker cooperatively and join it so no generation thread leaks.
            cancel.set()
            thread.join()

        if clean_exit and worker_error:
            raise worker_error[0]

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        prompt = self._build_prompt(messages, tools=tools)
        return self.generate(prompt, config)

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        prompt = self._build_prompt(messages, tools=tools)
        return self.generate_stream(prompt, config)

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
