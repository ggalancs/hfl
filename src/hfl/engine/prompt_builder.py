# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Unified prompt building for different chat formats.

Centralizes prompt construction logic used by multiple engines.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hfl.engine.base import ChatMessage


class PromptFormat(Enum):
    """Supported chat prompt formats."""

    CHATML = "chatml"  # <|im_start|>role\ncontent<|im_end|>
    LLAMA2 = "llama2"  # [INST] content [/INST]
    LLAMA3 = "llama3"  # <|begin_of_text|><|start_header_id|>
    ALPACA = "alpaca"  # ### Instruction: / ### Response:
    VICUNA = "vicuna"  # USER: / ASSISTANT:
    GENERIC = "generic"  # Simple role: content format


class PromptBuilder:
    """Builds prompts for different chat formats.

    Centralizes prompt construction to avoid duplication across engines.
    """

    @staticmethod
    def _escape_content(content: str, format: PromptFormat) -> str:
        """Escape format-specific delimiters in user content.

        Prevents prompt injection via format delimiter manipulation.
        """
        if format == PromptFormat.CHATML:
            content = content.replace("<|im_start|>", r"\<|im_start|\>")
            content = content.replace("<|im_end|>", r"\<|im_end|\>")
        elif format == PromptFormat.LLAMA3:
            content = content.replace("<|begin_of_text|>", "")
            content = content.replace("<|start_header_id|>", "")
            content = content.replace("<|end_header_id|>", "")
            content = content.replace("<|eot_id|>", "")
        elif format == PromptFormat.LLAMA2:
            content = content.replace("[INST]", "")
            content = content.replace("[/INST]", "")
            content = content.replace("<<SYS>>", "")
            content = content.replace("<</SYS>>", "")
        elif format == PromptFormat.ALPACA:
            content = content.replace("### Instruction:", "")
            content = content.replace("### Response:", "")
            content = content.replace("### System:", "")
        return content

    @staticmethod
    def messages_to_dicts(messages: list["ChatMessage"]) -> list[dict[str, str]]:
        """Convert ChatMessage objects to dictionaries.

        Args:
            messages: List of ChatMessage objects

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return [{"role": m.role, "content": m.content} for m in messages]

    @staticmethod
    def build_chatml(
        messages: list["ChatMessage"],
        add_generation_prompt: bool = True,
    ) -> str:
        """Build prompt in ChatML format.

        Used by: ChatGPT, many OpenAI-style models

        Args:
            messages: List of chat messages
            add_generation_prompt: Whether to add assistant start token

        Returns:
            Formatted prompt string
        """
        parts = []
        for msg in messages:
            escaped = PromptBuilder._escape_content(msg.content, PromptFormat.CHATML)
            parts.append(f"<|im_start|>{msg.role}\n{escaped}<|im_end|>")

        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    @staticmethod
    def build_llama2(
        messages: list["ChatMessage"],
        add_generation_prompt: bool = True,
    ) -> str:
        """Build prompt in Llama 2 format.

        Used by: Llama 2, Mistral, many instruction-tuned models

        Args:
            messages: List of chat messages
            add_generation_prompt: Whether to add space for response

        Returns:
            Formatted prompt string
        """
        parts = []
        system_content = ""

        for msg in messages:
            escaped = PromptBuilder._escape_content(msg.content, PromptFormat.LLAMA2)
            if msg.role == "system":
                system_content = escaped
            elif msg.role == "user":
                if system_content:
                    parts.append(f"[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{escaped} [/INST]")
                    system_content = ""
                else:
                    parts.append(f"[INST] {escaped} [/INST]")
            elif msg.role == "assistant":
                parts.append(escaped)

        return " ".join(parts)

    @staticmethod
    def build_llama3(
        messages: list["ChatMessage"],
        add_generation_prompt: bool = True,
    ) -> str:
        """Build prompt in Llama 3 format.

        Used by: Llama 3, Llama 3.1, Llama 3.2

        Args:
            messages: List of chat messages
            add_generation_prompt: Whether to add assistant start token

        Returns:
            Formatted prompt string
        """
        parts = ["<|begin_of_text|>"]

        for msg in messages:
            escaped = PromptBuilder._escape_content(msg.content, PromptFormat.LLAMA3)
            parts.append(f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{escaped}<|eot_id|>")

        if add_generation_prompt:
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(parts)

    @staticmethod
    def build_alpaca(
        messages: list["ChatMessage"],
        add_generation_prompt: bool = True,
    ) -> str:
        """Build prompt in Alpaca format.

        Used by: Alpaca, many early instruction-tuned models

        Args:
            messages: List of chat messages
            add_generation_prompt: Whether to add response prompt

        Returns:
            Formatted prompt string
        """
        parts = []

        for msg in messages:
            escaped = PromptBuilder._escape_content(msg.content, PromptFormat.ALPACA)
            if msg.role == "system":
                parts.append(f"### System:\n{escaped}\n")
            elif msg.role == "user":
                parts.append(f"### Instruction:\n{escaped}\n")
            elif msg.role == "assistant":
                parts.append(f"### Response:\n{escaped}\n")

        if add_generation_prompt:
            parts.append("### Response:\n")

        return "\n".join(parts)

    @staticmethod
    def build_vicuna(
        messages: list["ChatMessage"],
        add_generation_prompt: bool = True,
    ) -> str:
        """Build prompt in Vicuna format.

        Used by: Vicuna, some ShareGPT-trained models

        Args:
            messages: List of chat messages
            add_generation_prompt: Whether to add assistant prompt

        Returns:
            Formatted prompt string
        """
        parts = []

        for msg in messages:
            if msg.role == "system":
                parts.append(msg.content)
            elif msg.role == "user":
                parts.append(f"USER: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"ASSISTANT: {msg.content}")

        if add_generation_prompt:
            parts.append("ASSISTANT:")

        return "\n\n".join(parts)

    @staticmethod
    def build_generic(
        messages: list["ChatMessage"],
        add_generation_prompt: bool = True,
    ) -> str:
        """Build prompt in generic role: content format.

        Simple fallback format for unknown models.

        Args:
            messages: List of chat messages
            add_generation_prompt: Whether to add assistant prompt

        Returns:
            Formatted prompt string
        """
        parts = []

        for msg in messages:
            role_name = msg.role.capitalize()
            parts.append(f"{role_name}: {msg.content}")

        if add_generation_prompt:
            parts.append("Assistant:")

        return "\n\n".join(parts)

    @staticmethod
    def _prepend_tools_system(
        messages: list["ChatMessage"],
        tools: list[dict] | None,
    ) -> list["ChatMessage"]:
        """Return ``messages`` with a synthetic system preamble that lists
        the available tools.

        Used by backends that do not have a tool-aware chat template
        (raw ``PromptBuilder`` formats). The model output is still parsed
        upstream by :mod:`hfl.api.tool_parsers`; this preamble is what
        gives the model a chance to emit a recognizable call envelope.
        """
        if not tools:
            return messages
        import json as _json

        from hfl.engine.base import ChatMessage as _CM

        lines = [
            "You have access to the following tools. When you need to call "
            "one, respond with a single JSON object of the form "
            '{"name": "<tool_name>", "arguments": {...}} and nothing else.',
            "Tools:",
            _json.dumps([t for t in tools], ensure_ascii=False),
        ]
        preamble = _CM(role="system", content="\n".join(lines))
        return [preamble, *messages]

    @classmethod
    def build(
        cls,
        messages: list["ChatMessage"],
        format: PromptFormat = PromptFormat.GENERIC,
        add_generation_prompt: bool = True,
        tools: list[dict] | None = None,
    ) -> str:
        """Build prompt using specified format.

        Args:
            messages: List of chat messages
            format: Prompt format to use
            add_generation_prompt: Whether to add generation prompt
            tools: Optional list of tool definitions. When provided, a
                best-effort preamble is injected so backends without a
                tool-aware chat template still have a chance of emitting
                a recognizable tool-call envelope.

        Returns:
            Formatted prompt string
        """
        builders = {
            PromptFormat.CHATML: cls.build_chatml,
            PromptFormat.LLAMA2: cls.build_llama2,
            PromptFormat.LLAMA3: cls.build_llama3,
            PromptFormat.ALPACA: cls.build_alpaca,
            PromptFormat.VICUNA: cls.build_vicuna,
            PromptFormat.GENERIC: cls.build_generic,
        }

        effective_messages = cls._prepend_tools_system(messages, tools)
        builder = builders.get(format, cls.build_generic)
        return builder(effective_messages, add_generation_prompt)


def detect_format_from_tokenizer(tokenizer) -> PromptFormat:
    """Detect prompt format from tokenizer configuration.

    Args:
        tokenizer: Transformers tokenizer instance

    Returns:
        Detected PromptFormat
    """
    # Check chat template
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        template = tokenizer.chat_template.lower()
        if "im_start" in template:
            return PromptFormat.CHATML
        if "begin_of_text" in template or "start_header_id" in template:
            return PromptFormat.LLAMA3
        if "[inst]" in template:
            return PromptFormat.LLAMA2

    # Check model name in tokenizer config
    if hasattr(tokenizer, "name_or_path"):
        name = tokenizer.name_or_path.lower()
        if "llama-3" in name or "llama3" in name:
            return PromptFormat.LLAMA3
        if "llama-2" in name or "llama2" in name:
            return PromptFormat.LLAMA2
        if "vicuna" in name:
            return PromptFormat.VICUNA
        if "alpaca" in name:
            return PromptFormat.ALPACA

    return PromptFormat.GENERIC
