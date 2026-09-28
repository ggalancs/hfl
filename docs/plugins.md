# Engine plugins

An engine runs models: HFL ships llama.cpp, llama-server, MLX,
Transformers and vLLM. Another package can add one, and HFL serves models
with it like its own: every API, streaming, the queue, keep-alive.

A working example is in
[`examples/hfl-echo-engine`](../examples/hfl-echo-engine): an engine that
answers with the last message. It was installed and served for real
(`hfl serve --backend echo` answered, and `/api/ps` named its engine).

## Write one

1. Subclass `hfl.engine.base.InferenceEngine` and implement `load`,
   `unload`, `generate`, `generate_stream`, `chat`, `chat_stream` and the
   `model_name` / `is_loaded` properties. `generate` and `chat` return a
   `GenerationResult` (`text`, token counts); the `_stream` methods yield
   text pieces.
2. Register the class under the `hfl.engines` entry point in the package's
   `pyproject.toml`:

   ```toml
   [project.entry-points."hfl.engines"]
   echo = "hfl_echo_engine:EchoEngine"
   ```

3. Install the package where HFL is installed (`pip install ./my-engine`).

## Use it

```bash
hfl serve --backend echo          # this server, every model
HFL_LLM_LIBRARY=echo hfl serve    # the same, from the environment
```

An unknown name is refused, with the installed engines listed. A plugin
cannot take a built-in engine's name (`llama-cpp`, `llama-server`, `mlx`,
`transformers`, `vllm`): choosing one of those never runs a plugin's
code. A plugin whose entry point does not produce an `InferenceEngine` is
refused when it is chosen.

A plugin runs inside the HFL server with its permissions: install only
plugins you trust, as with any Python package.

Text-to-speech engines have an entry point too (`hfl.tts_engines`), but
nothing reads it yet: a TTS plugin is not used.
