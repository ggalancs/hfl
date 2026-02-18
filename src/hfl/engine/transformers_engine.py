# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Backend basado en HuggingFace Transformers.

Usa el modelo en su formato nativo (safetensors) con GPU.
Soporta cuantización dinámica via bitsandbytes.
"""

import time
from typing import Iterator

from hfl.engine.base import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
)


class TransformersEngine(InferenceEngine):
    """Motor de inferencia HuggingFace Transformers."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_id = ""

    def load(self, model_path: str, **kwargs) -> None:
        """
        Carga un modelo desde directorio local o repo HF.

        Args:
            model_path: Ruta local o repo_id de HuggingFace
            **kwargs:
                quantization: "4bit", "8bit", None
                device_map: "auto" (default), "cpu", "cuda:0"
                torch_dtype: "auto", "float16", "bfloat16"
                max_memory: dict de memoria máxima por dispositivo
                trust_remote_code: bool (default False)
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant = kwargs.get("quantization")
        load_kwargs = {
            "device_map": kwargs.get("device_map", "auto"),
            "torch_dtype": kwargs.get("torch_dtype", "auto"),
            "trust_remote_code": kwargs.get("trust_remote_code", False),
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

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self._model_id = model_path

    def unload(self) -> None:
        if self._model:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _build_prompt(self, messages: list[ChatMessage]) -> str:
        """Construye el prompt usando el chat template del tokenizer."""
        msgs = [{"role": m.role, "content": m.content} for m in messages]

        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback genérico
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"<<SYS>>{m.content}<</SYS>>")
            elif m.role == "user":
                parts.append(f"[INST] {m.content} [/INST]")
            elif m.role == "assistant":
                parts.append(m.content)
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

        # Decodificar solo los tokens nuevos
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
        from threading import Thread

        from transformers import TextIteratorStreamer

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
            "do_sample": cfg.temperature > 0,
            "streamer": streamer,
        }

        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        for text in streamer:
            if text:
                yield text

        thread.join()

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        prompt = self._build_prompt(messages)
        return self.generate(prompt, config)

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        prompt = self._build_prompt(messages)
        return self.generate_stream(prompt, config)

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
