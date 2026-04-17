# Plan de Paridad Ollama → HFL

**Fecha:** 2026-04-17
**Versión base HFL:** 0.3.5
**Versión Ollama comparada:** rama `main` (docs.ollama.com, abril 2026)
**Autor:** Análisis comparativo completo, orientado a implementación

---

## 1. Metodología

1. **Superficie Ollama** — enumerada desde `docs.ollama.com/api` (16 endpoints REST), `docs.ollama.com/cli` (subcomandos CLI),
   `docs.ollama.com/modelfile` (formato declarativo) y `docs.ollama.com/capabilities/structured-outputs`.
2. **Superficie HFL** — grep de `@router.*` en `src/hfl/api/*` y `@app.command` en `src/hfl/cli/main.py`, complementado con el
   CHANGELOG hasta 0.3.5.
3. **Criterio de "paridad"** — un feature de Ollama se considera cubierto en HFL si existe el mismo endpoint / comando / campo,
   con la misma semántica observable por un cliente. Compatibilidad binaria exacta del body JSON es objetivo cuando la ruta
   lleva el prefijo `/api/` (namespace Ollama-native); para el namespace `/v1/` la semántica de OpenAI manda.
4. **Scope** — "mejoras que sirvan para HFL". Los aspectos cerrados/comerciales de Ollama (signin, launch de integraciones
   propietarias) no son candidatos; se listan pero se marcan *no aplicables*.

---

## 2. Matriz comparativa resumida

### 2.1 Endpoints REST

| # | Ollama | Método + path | HFL equivalente | Estado | Prioridad |
|---|---|---|---|---|---|
|  1 | Generate completion | `POST /api/generate` | idem | ✅ Completo | — |
|  2 | Chat completion | `POST /api/chat` | idem | ✅ Completo | — |
|  3 | Create model | `POST /api/create` | — | ❌ Falta | P2 |
|  4 | List local models | `GET /api/tags` | idem | ✅ Completo | — |
|  5 | Show model info | `POST /api/show` | — | ❌ Falta | **P0** |
|  6 | Copy model | `POST /api/copy` | `hfl alias` (solo CLI) | 🟡 Parcial | P1 |
|  7 | Delete model | `DELETE /api/delete` | idem | ✅ Completo | — |
|  8 | Pull model | `POST /api/pull` | idem | 🟡 Revisar formato progress | P1 |
|  9 | Push model | `POST /api/push` | — | ❌ Falta (requiere registry) | P3 |
| 10 | Generate embeddings | `POST /api/embed` | — | ❌ Falta | **P0** |
| 11 | List running models | `GET /api/ps` | — | ❌ Falta | **P0** |
| 12 | Check blob | `HEAD /api/blobs/:digest` | — | ❌ Falta (requiere create) | P2 |
| 13 | Push blob | `POST /api/blobs/:digest` | — | ❌ Falta (requiere create) | P2 |
| 14 | Get version | `GET /api/version` | idem | ✅ Completo | — |
| 15 | Embeddings legacy | `POST /api/embeddings` | — | ❌ Falta | P1 (alias de `/api/embed`) |
| 16 | OpenAI `/v1/embeddings` | — | — | ❌ Falta | **P0** (para paridad OpenAI) |

### 2.2 Parámetros de request (contenido)

| Campo | Ollama (`/api/chat`, `/api/generate`) | HFL | Estado |
|---|---|---|---|
| `model` | ✅ | ✅ | Paridad |
| `prompt` / `messages` | ✅ | ✅ | Paridad |
| `stream` | ✅ | ✅ | Paridad |
| `tools` | ✅ | ✅ (qwen/llama3/gemma4/mistral) | Paridad |
| `options` | ✅ (temperature, top_p, seed, num_predict, num_ctx, repeat_penalty, etc.) | ✅ (subset) | Paridad |
| `images` (vision) | ✅ (base64) | ❌ | **P0** |
| `format` = `"json"` | ✅ | ❌ | **P0** |
| `format` = `<json-schema>` | ✅ | ❌ | **P0** |
| `keep_alive` | ✅ (string duration, default "5m") | ❌ | **P0** |
| `system` (override) | ✅ | ❌ | P1 |
| `template` (override) | ✅ | ❌ | P2 |
| `raw` (disable template) | ✅ | ❌ | P2 |
| `suffix` (infilling) | ✅ | ❌ | P3 |
| `think` (reasoning) | ✅ (nuevo en 2026) | 🟡 (parsers de Gemma4/DeepSeek) | P1 — exponer flag |
| `context` (legacy) | 🟡 deprecated | ❌ | No aplicable |
| `dimensions` (embed) | ✅ (Matryoshka) | ❌ | P1 (depende de P0 embed) |
| `truncate` (embed) | ✅ | ❌ | P1 |

### 2.3 Response fields

| Campo final (non-streaming) | Ollama | HFL | Estado |
|---|---|---|---|
| `total_duration` | ✅ | 🟡 (algunos endpoints) | P2 (uniformar) |
| `load_duration` | ✅ | ❌ | P2 |
| `prompt_eval_count` | ✅ | ✅ (`tokens_prompt`) | Paridad semántica |
| `prompt_eval_duration` | ✅ | ❌ | P2 |
| `eval_count` | ✅ | ✅ | Paridad |
| `eval_duration` | ✅ | ❌ | P2 |
| `context` (array) | 🟡 deprecated | ❌ | No aplicable |

### 2.4 CLI

| Ollama CLI | HFL CLI | Estado |
|---|---|---|
| `ollama run <model>` | `hfl run <model>` | ✅ |
| `ollama pull <model>` | `hfl pull <model>` | ✅ |
| `ollama list` (`ls`) | `hfl list` | ✅ |
| `ollama rm <model>` | `hfl rm <model>` | ✅ |
| `ollama serve` | `hfl serve` | ✅ |
| `ollama ps` | — | ❌ **P0** |
| `ollama stop <model>` | — | ❌ **P0** |
| `ollama show <model>` | `hfl inspect <model>` | 🟡 (reformatear a paridad) |
| `ollama cp <src> <dst>` | `hfl alias <model> <name>` | 🟡 (copy vs. alias distinto) |
| `ollama create <name> -f Modelfile` | — | ❌ P2 |
| `ollama push <model>` | — | ❌ P3 |
| `ollama signin / signout` | — | N/A (servicio propietario) |
| `ollama launch <integration>` | — | N/A (wrapper propietario) |

### 2.5 Modelfile (formato declarativo)

| Instrucción | Ollama | HFL | Estado |
|---|---|---|---|
| `FROM` | ✅ | 🟡 (implícito en manifest.json) | P2 — DSL texto |
| `PARAMETER <k> <v>` | ✅ (num_ctx, temperature, top_k, top_p, min_p, repeat_last_n, repeat_penalty, seed, stop, num_predict) | 🟡 (GenerationConfig en request) | P2 — baked-in defaults |
| `TEMPLATE` | ✅ | ❌ | P2 |
| `SYSTEM` | ✅ | ❌ | P2 |
| `ADAPTER` | ✅ (LoRA) | ❌ | P3 |
| `LICENSE` | ✅ | 🟡 (manifest tiene `license_id`) | P2 |
| `MESSAGE` (few-shot) | ✅ | ❌ | P3 |
| `REQUIRES` (versión mínima) | ✅ | ❌ | P3 |

---

## 3. Gaps detallados y plan de implementación

### P0-1 — Endpoint de embeddings (`/api/embed` + `/v1/embeddings`)

**Impacto:** Bloqueante para RAG, búsqueda semántica, y cualquier pipeline que use HFL como drop-in de Ollama en apps
LangChain / LlamaIndex. Hoy un cliente que haga `embed` contra HFL recibe 404.

**Alcance técnico:**

- Nuevo módulo `src/hfl/engine/embedding_engine.py` con interfaz `EmbeddingEngine(InferenceEngine)` y método
  `embed(inputs: list[str], *, truncate: bool, dimensions: int | None) -> list[list[float]]`.
- Implementación sobre llama-cpp-python (`Llama(embedding=True)` + `embed()`) y sentence-transformers como fallback (
  requiere nuevo extra `[embed]` en `pyproject.toml`).
- Detección de modelo: `detect_model_type` amplía el enum `ModelType` con `EMBED`; GGUFs con `general.architecture` en
  `{bert, nomic-bert, jina-bert}` se clasifican automáticamente como embed.
- Rutas:
  - `POST /api/embed` (body: `{model, input, truncate=true, options, keep_alive, dimensions}`) — Ollama-native.
  - `POST /api/embeddings` (body legacy: `{model, prompt, options, keep_alive}`) — alias deprecado.
  - `POST /v1/embeddings` (body OpenAI: `{model, input, encoding_format, dimensions, user}`) — OpenAI-compatible.
- Dispatcher: embeddings **no** van por la misma cola que chat — son stateless, paralelizables. Nuevo `EmbeddingDispatcher`
  con `max_concurrent` propio (default 4) y métricas separadas.
- Prometheus: `hfl_embedding_requests_total{model,status}`, `hfl_embedding_latency_seconds`.

**Sub-tareas:**

1. Implementar `EmbeddingEngine` base y `LlamaCppEmbeddingEngine`.
2. Implementar `SentenceTransformersEmbeddingEngine` bajo extra `[embed]`.
3. Añadir `ModelType.EMBED` y detección por metadata GGUF.
4. Rutas Ollama `/api/embed` + `/api/embeddings` (legacy).
5. Ruta OpenAI `/v1/embeddings` con serialización compatible.
6. `EmbeddingDispatcher` con cola separada + métricas.
7. Tests unitarios (20+): shape del response, truncation, dimensions (Matryoshka), modelos no-embed → 400, batching.
8. Tests de integración con un GGUF real pequeño (nomic-embed-text-v1.5 Q4_K_M ~80 MB) guardado en un fixture.

**Archivos tocados (estimado):** 12 archivos, ~900 LOC, ~40 tests nuevos.

---

### P0-2 — `/api/ps` y `hfl ps` (modelos en memoria)

**Impacto:** Operacional crítico. Sin esto, un operador no puede ver qué modelos están residentes, cuánto ocupan, ni cuándo
expiran. Clientes que usan Ollama SDK esperan este endpoint para UI tipo `Open WebUI`.

**Alcance técnico:**

- Aprovechar `ModelPool._models` (ya es `OrderedDict[str, CachedModel]` con `last_used` y `memory_estimate_mb`) —
  añadir `expires_at` basado en `idle_timeout_seconds`.
- Ruta `GET /api/ps` devuelve:
  ```json
  {
    "models": [
      {
        "name": "qwen-coder:7b",
        "model": "qwen-coder:7b",
        "size": 4200000000,
        "digest": "sha256:...",
        "details": {"format": "gguf", "family": "qwen", "parameter_size": "7B", "quantization_level": "Q4_K_M"},
        "expires_at": "2026-04-17T15:30:00Z",
        "size_vram": 4100000000
      }
    ]
  }
  ```
- `hfl ps` CLI: tabla con columnas NAME / ID / SIZE / PROCESSOR / UNTIL.
- VRAM vs. RAM: en Mac/Metal el `size_vram` se infiere de `metal_memory_estimate`; en CPU es 0.

**Sub-tareas:**

1. Añadir `expires_at` calculado en `CachedModel.to_dict()`.
2. Nuevo módulo `src/hfl/api/routes_ps.py` con `GET /api/ps`.
3. Registrar el router en `server.py`.
4. Extender `state.py` para exponer una vista read-only del pool sin lock contention.
5. `hfl ps` CLI con formato tipo `ollama ps`.
6. Tests: pool vacío, un modelo, dos modelos, uno con VRAM otro en CPU, expired-pero-todavía-listed.

**Archivos tocados:** 5 archivos, ~250 LOC, 10 tests.

---

### P0-3 — `/api/show` y `hfl show`

**Impacto:** Clientes que inspeccionan capacidades (p.ej. para decidir si activar tool-calling) dependen del campo
`capabilities` de `/api/show`. Sin él, `ollama-python` devuelve `UNKNOWN` a algunas llamadas.

**Alcance técnico:**

- Ruta `POST /api/show` body `{model, verbose?}` → respuesta:
  ```json
  {
    "modelfile": "...",          // texto Modelfile renderizado desde el manifest
    "parameters": "temperature 0.7\nstop \"<|im_end|>\"",  // formato Ollama
    "template": "<Jinja template>",
    "details": {"format":"gguf","family":"qwen","parameter_size":"7B","quantization_level":"Q4_K_M"},
    "model_info": {
      "general.architecture":"qwen",
      "general.parameter_count": 7_615_616_512,
      // ... toda la metadata GGUF
    },
    "capabilities": ["completion","tools","insert","vision","embedding","thinking"]
  }
  ```
- `capabilities` se infiere:
  - `completion` — siempre para LLMs.
  - `tools` — modelo en la tabla `_FAMILIES_WITH_TOOL_SUPPORT` de `tool_parsers.py`.
  - `insert` — FIM / suffix support (codellama, codegemma, starcoder, qwen-coder).
  - `vision` — metadata GGUF declara `clip.*` o arquitectura en `{llava, gemma3-v, qwen2-vl, ...}`.
  - `embedding` — `ModelType.EMBED`.
  - `thinking` — modelos con canal de reasoning (gemma4, deepseek-r1, qwen3-thinking, gpt-oss).
- `hfl show <model>` — imprime el Modelfile renderizado + capabilities.

**Sub-tareas:**

1. Nuevo módulo `src/hfl/converter/modelfile.py`: `render_modelfile(manifest: ModelManifest) -> str`.
2. Detector de capabilities en `src/hfl/models/capabilities.py`.
3. Ruta `POST /api/show`.
4. `hfl show` CLI.
5. Tests: cada capability detectado por un GGUF de muestra (mock de metadata).

**Archivos tocados:** 6 archivos, ~400 LOC, 18 tests.

---

### P0-4 — `keep_alive` por request

**Impacto:** Multi-tenant real. Hoy HFL tiene `ModelPool.idle_timeout_seconds` global; un cliente no puede decir "quiero
este modelo tibio 30 min más" en un request específico.

**Alcance técnico:**

- Parseo del valor Ollama-style: `"5m"`, `"30s"`, `"1h30m"`, `-1` (infinito), `0` (unload inmediato tras la request).
- Nueva llave en `CachedModel`: `keep_alive_deadline: datetime | None` — cuando está set, domina sobre `idle_timeout`.
- En cada request (chat / generate / embed) extraer `keep_alive` de body y, si viene, `pool.set_keep_alive(model, duration)`.
- Implementar `keep_alive=0` → `pool.evict(model)` tras responder.
- Documentar interacción con `queue_max_inflight=1`: el unload espera a que el slot esté libre.

**Sub-tareas:**

1. Parser `parse_duration(str) -> timedelta | None` en `src/hfl/utils/duration.py`.
2. Extensión de `ModelPool` con `set_keep_alive(name, duration)` y `evict_after(name, deadline)`.
3. Integración en las 3 rutas (chat/generate/embed).
4. CLI flag `hfl run --keep-alive 10m`.
5. Tests (mínimo 15): todos los formatos de duración, keep_alive=0, keep_alive=-1, interacción con `idle_timeout`, race
   entre `keep_alive=0` y queue pendiente.

**Archivos tocados:** 8 archivos, ~500 LOC, 15+ tests.

---

### P0-5 — Structured outputs (`format=json` y `format=<schema>`)

**Impacto:** Clientes de tool-calling avanzado (LangChain, LlamaIndex, instructor) dependen de esto. Es un feature de
**generación restringida** (GBNF / Outlines) — obliga al modelo a emitir sólo tokens que respeten la gramática.

**Alcance técnico:**

- **Vía llama.cpp:** `llama-cpp-python` acepta `grammar=LlamaGrammar.from_json_schema(schema)`. Implementar:
  - `format="json"` → `grammar = LlamaGrammar.from_string(JSON_ROOT_GBNF)` (gramática ya integrada en llama-cpp).
  - `format={...schema...}` → `grammar = LlamaGrammar.from_json_schema(json.dumps(schema))`.
- **Vía Transformers:** usar `outlines` (extra nuevo `[outlines]`) o `lm-format-enforcer` — guided generation.
- **Vía vLLM:** vLLM tiene `GuidedDecodingParams(json=schema)` nativo desde 0.5+.
- Validación del schema en el router: rechazar schemas recursivos sin límite (prevención DoS: `max_depth=10`,
  `max_properties=200`).
- Ruta OpenAI `/v1/chat/completions`: aceptar `response_format: {"type":"json_object"}` y
  `response_format: {"type":"json_schema", "json_schema":{...}}` (paridad OpenAI JSON Mode).
- Ruta Ollama `/api/chat` y `/api/generate`: aceptar campo `format`.

**Sub-tareas:**

1. Añadir `grammar: str | None` y `response_format: dict | None` a `GenerationConfig`.
2. En `LlamaCppEngine.chat/generate`: si `grammar` está, pasárselo a `Llama.create_chat_completion(grammar=...)`.
3. En `TransformersEngine`: extra opcional `[outlines]`, fallback a regex post-hoc si no está instalado.
4. En `VLLMEngine`: mapear a `GuidedDecodingParams`.
5. Validador de schema (`src/hfl/validators.py::validate_json_schema`).
6. Schemas compatibles con OpenAI en `api/schemas/openai.py` y Ollama en `api/schemas/ollama.py`.
7. Tests: 25+ — JSON vacío, schema simple, schema anidado, schema inválido → 400, schema recursivo → 400, integración real
   con un modelo pequeño, comparación de output con/sin grammar.

**Archivos tocados:** 14 archivos, ~1100 LOC, 25+ tests.

---

### P0-6 — Multi-modal / visión (imágenes en chat)

**Impacto:** Modelos de visión (Llama 3.2 Vision, Gemma 3 multimodal, Qwen2-VL, InternVL, LLaVA) dominan el espacio de
agentes visuales y RAG con diagramas. Sin soporte, HFL queda fuera de este segmento.

**Alcance técnico:**

- **Ollama-native:** `messages[i].images: list[str]` (base64 PNG/JPEG). Decodificar y pasar al engine.
- **OpenAI-compatible:** `content: [{"type":"text",...}, {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]`.
- **llama-cpp-python:** requiere `clip_model_path` adicional → extender manifest con `clip_model_path` opcional y
  `Llama(clip_model_path=...)`.
- **Transformers:** `AutoModelForVision2Seq` — añadir selector en `engine/selector.py`.
- **vLLM:** soporta visión nativamente en 0.5+ vía `MultimodalData`.
- Validación de tamaño: límite de 20 MB por imagen (body-limit ya existe); rechazar > 4096×4096.
- Validación de MIME: whitelist `{image/png, image/jpeg, image/webp}`.

**Sub-tareas:**

1. `ChatMessage` soporta `images: list[bytes] | None`.
2. `LlamaCppEngine`: cargar CLIP projector desde el manifest si está presente.
3. `TransformersEngine`: detectar modelo vision, cargar `AutoProcessor` + `AutoModelForVision2Seq`.
4. Conversión OpenAI `content: [...]` → `ChatMessage.images` en `api/converters.py`.
5. Validador `validate_image(bytes) -> bytes` (MIME + dimensiones + magic bytes anti-SVG/anti-script).
6. Schema: `ChatCompletionMessage.content: str | list[ContentPart]` (union).
7. Tests (30+): PNG válido, JPEG válido, SVG rechazado, > 20 MB rechazado, dimensiones > 4K rechazado, modelo no-vision
   con imagen → 400, integración real con gemma-3-4b-it-mmproj.

**Archivos tocados:** 16 archivos, ~1400 LOC, 30+ tests.

---

### P1-1 — `system` y `think` en requests

**Impacto:** Clientes que quieren sobreescribir el system prompt del Modelfile por request (p.ej. agents framework con
personalidades rotantes) hoy están bloqueados.

**Alcance:**

- `generate` y `chat` aceptan `system: str` — si está presente, se inyecta como primer `ChatMessage(role="system")`
  reemplazando el system del manifest.
- `think: bool` — expone el canal de razonamiento en lugar de filtrarlo. Hoy `_Gemma4StreamFilter` lo ELIMINA siempre.
  Si `think=true`, el filtro deja pasar el canal pero los eventos van a `message.thinking` en vez de `message.content`
  (paridad con Ollama 2026).

**Sub-tareas:**

1. Añadir `system`, `think` a los schemas de `/api/generate` y `/api/chat`.
2. En `routes_native.py`, construir la lista de messages tras el system override.
3. En `LlamaCppEngine`, flag `expose_reasoning: bool` que cambia el comportamiento de los channel filters.
4. Tests: 10+ — system override funciona, think=true para gemma4, think=false para gemma4 (default), modelo sin thinking
   ignora think.

**Archivos tocados:** 7 archivos, ~350 LOC, 10 tests.

---

### P1-2 — `/api/copy`, `hfl cp`, diferenciación con alias

**Impacto:** UX Ollama-idéntica para scripts que esperan `POST /api/copy {"source":"qwen-7b","destination":"my-qwen"}`.

**Alcance:**

- Hoy `hfl alias` crea un *apuntador* (misma entry `local_path`). `ollama cp` **duplica** el manifest (distinto digest, misma
  blob subyacente via hard-link / ref-count).
- Implementación:
  - Nuevo método `ModelRegistry.copy(src_name: str, dst_name: str)` que hace `copy.deepcopy(manifest)` + nuevo `digest`
    (sha256 del manifest serializado, no del blob).
  - No duplicar los archivos del modelo — ref-count por `local_path`.
  - Tras delete, decrementar ref-count; sólo si llega a 0, borrar la blob.
- Ruta `POST /api/copy` body `{source, destination}` → 200 OK o 404.
- `hfl cp <src> <dst>` CLI (mantener `hfl alias` como shortcut aparte).

**Sub-tareas:**

1. Añadir `blob_ref_count` al schema del registry (migración compatible: default 1).
2. `ModelRegistry.copy()` + `delete()` con decremento.
3. Ruta `/api/copy`.
4. `hfl cp` CLI.
5. Tests: copy + delete de una, delete de la otra, delete de ambas libera blob.

**Archivos tocados:** 6 archivos, ~300 LOC, 12 tests.

---

### P1-3 — Métricas de rendimiento uniformes en todos los responses

**Impacto:** Clientes que miden tokens/s (uso interno, billing, monitoring) dependen de `eval_duration`, `load_duration`,
`total_duration`. Hoy HFL devuelve `0` en muchos campos.

**Alcance:**

- Timing granular en `LlamaCppEngine.generate/chat`:
  - `load_duration` = tiempo de primer token (si modelo acababa de cargarse).
  - `prompt_eval_duration` = tiempo de procesar el prompt (pre-primer-token).
  - `eval_duration` = tiempo de generar tokens.
  - `total_duration` = suma de los tres.
- Todos en **nanosegundos** (convención Ollama).
- `GenerationResult` crece con los 4 campos.

**Sub-tareas:**

1. Instrumentar `LlamaCppEngine` con `time.monotonic_ns()` en los 3 puntos.
2. Hacer lo mismo en `TransformersEngine` y `VLLMEngine` (más difícil: Transformers no expone el momento del primer token
   sin un callback).
3. Serializar los 4 campos en los 3 namespaces (Ollama, OpenAI adapter, Anthropic adapter).
4. Tests: valores no-cero tras una generación real; relación `total = load + prompt_eval + eval` ± 10 ms tolerance.

**Archivos tocados:** 9 archivos, ~400 LOC, 12 tests.

---

### P1-4 — `hfl stop <model>` + endpoint `DELETE` del pool

**Impacto:** Operador quiere liberar VRAM antes de cargar otro modelo sin reiniciar el server.

**Alcance:**

- CLI `hfl stop <model>` — hace `POST /api/chat` con body `{"model":"X","keep_alive":0,"messages":[]}` (Ollama trick) o
  llama a un nuevo endpoint `DELETE /api/running/:model`.
- Implementación limpia: exponer `pool.evict(model)` vía `POST /api/stop` body `{"model":"X"}` o con el truco del
  `keep_alive=0` + messages vacíos.
- Paridad Ollama: ambos funcionan.

**Sub-tareas:**

1. Interpretar `keep_alive=0` + `messages=[]` como "sólo carga/descarga" y devolver vacío.
2. Implementar `messages=[]` → short-circuit: sólo gestión de pool.
3. `hfl stop` CLI.
4. Tests: stop de modelo cargado → expulsado del pool; stop de modelo no cargado → 200 sin cambios.

**Archivos tocados:** 5 archivos, ~200 LOC, 8 tests.

---

### P1-5 — Progress streaming de `pull` con campos Ollama-idénticos

**Impacto:** Open WebUI y otras UIs esperan NDJSON con campos `status`, `digest`, `total`, `completed`. Si HFL devuelve
otra estructura, la barra de progreso nunca aparece.

**Alcance:**

- Inspeccionar la implementación actual de `/api/pull` en HFL y compararla con la de Ollama.
- Ollama emite por layer: `{"status":"pulling manifest"}`, `{"status":"pulling <digest>","digest":"sha256:...","total":N,"completed":M}` cada ~100 ms,
  y al final `{"status":"success"}`.
- HFL debe seguir ese patrón.

**Sub-tareas:**

1. Auditar `hub/downloader.py` y `routes_native.py::pull_model`.
2. Emitir exactamente ese JSON por chunk.
3. Rate-limitar el progress a 10 Hz máximo para no inundar.
4. Tests: mock hub descarga de 3 layers → NDJSON válido con los 5 statuses esperados.

**Archivos tocados:** 4 archivos, ~250 LOC, 8 tests.

---

### P2-1 — Soporte de Modelfile (`/api/create` + `hfl create`)

**Impacto:** Feature emblemática de Ollama. Permite al usuario crear variantes de un modelo con un system prompt, unos
parámetros por defecto y una plantilla, empaquetadas como "modelo propio".

**Alcance:**

- Parser de Modelfile en `src/hfl/converter/modelfile.py`:
  - Tokenizer línea a línea, instrucciones `FROM`, `PARAMETER`, `TEMPLATE`, `SYSTEM`, `ADAPTER`, `LICENSE`, `MESSAGE`,
    `REQUIRES`.
  - `FROM` acepta: nombre de modelo existente, path local a GGUF, path a directorio safetensors, URL HuggingFace.
- Persistencia: el Modelfile se guarda en `models/<name>/Modelfile` y se renderiza de vuelta en `/api/show`.
- El nuevo modelo crea un manifest derivado con `parent_digest` apuntando al FROM.
- `POST /api/create` acepta payload multipart con el Modelfile como body (o usando blobs pre-subidos).
- `hfl create <name> -f path/to/Modelfile` CLI.

**Sub-tareas:**

1. Parser + validador del Modelfile (gramática formal).
2. Modelo de datos `ModelfileDocument` con serialización round-trip.
3. `/api/create` streaming con progress.
4. `hfl create` CLI.
5. Integración: Modelfile params se aplican por defecto en `GenerationConfig` al cargar el modelo.
6. Tests (40+): parser, todos los tipos de `FROM`, render round-trip, ejecución de PARAMETER, herencia de padre, errores
   de sintaxis, Modelfile sin FROM → 400.

**Archivos tocados:** 20 archivos, ~2000 LOC, 40+ tests.

---

### P2-2 — Blob API (`HEAD/POST /api/blobs/:digest`)

**Impacto:** Sólo útil junto con `/api/create` — sirve para uploadear el GGUF antes de crear el modelo. Sin `/api/create`
no aporta.

**Alcance:**

- Storage: `config.home_dir / "blobs" / "sha256-<digest>"`.
- `HEAD /api/blobs/:digest` — 200 si existe, 404 si no.
- `POST /api/blobs/:digest` — stream-upload, valida digest SHA256 al final; 201 si coincide, 400 si no.
- Rate-limit el `POST` (ya existe body size limit, pero streaming requiere tratamiento especial).
- Integración con `/api/create` — el `files: {...: digest}` del body se resuelve contra estos blobs.

**Sub-tareas:**

1. Nuevo módulo `src/hfl/hub/blobs.py`.
2. Ruta `HEAD /api/blobs/:digest`.
3. Ruta `POST /api/blobs/:digest` (StreamingResponse / Upload).
4. Validación SHA256 streaming con `hashlib.sha256(...).update()` por chunk.
5. Tests: upload happy path, digest mismatch → 400, digest existente → 200 en HEAD, digest inexistente → 404.

**Archivos tocados:** 5 archivos, ~400 LOC, 15 tests.

---

### P2-3 — `template` y `raw` en request

**Impacto:** Usuarios que quieren probar chat templates alternativos o desactivar el template del todo (para prompts de
evaluación tipo "complete this") dependen de estos flags.

**Alcance:**

- `template: str` en body de `/api/generate` — si presente, sustituye el Jinja template del engine para esta request.
- `raw: bool` en body de `/api/generate` — si `true`, el prompt se envía tal cual, sin aplicar template ni BOS.

**Sub-tareas:**

1. `GenerationConfig` gana `template_override: str | None`, `raw: bool`.
2. `LlamaCppEngine.generate` — si `raw`, usa `Llama.__call__` directo; si `template_override`, carga `ChatFormatter`
   efímero.
3. Schemas `/api/generate`.
4. Tests: template override produce distinto output; raw salta BOS; `raw=True, chat()` → ValidationError (no es chat).

**Archivos tocados:** 6 archivos, ~300 LOC, 10 tests.

---

### P2-4 — Response padronizado `context` legacy

**Impacto:** Clientes legacy Ollama < 0.1.17 dependen del array `context` para continuaciones. Baja demanda; opcional.

Sub-tareas y alcance mínimos: añadir `context: list[int]` al response non-streaming de `/api/generate` si `options.keep_context=true`
(no default por coste de memoria). 1 archivo, ~50 LOC, 3 tests. Considerarlo sólo si alguien lo pide.

---

### P3-1 — `/api/push` + soporte de registry propio

**Impacto:** Permitiría a HFL actuar como registry interno (equivalente a `registry.ollama.ai`). Es una funcionalidad de
servidor central; requiere diseñar autenticación, namespaces, quotas y storage distribuido. **Fuera de alcance para un
servidor local-first.**

**Recomendación:** Documentar como *non-goal*. Alternativa: integrar con HuggingFace Hub vía `huggingface_hub.upload_folder`.

---

### P3-2 — `ADAPTER` (LoRA) y `MESSAGE` (few-shot baked)

**Impacto:** LoRA es un win grande para fine-tuning ligero; few-shot baked permite empaquetar ejemplos canónicos de uso.
Depende de que `/api/create` esté estable primero.

- `ADAPTER` requiere soporte en el engine — llama-cpp-python acepta `lora_path`.
- `MESSAGE` es puramente declarativo: se renderiza en cada request como prefix de `messages`.

**Sub-tareas** (estimadas, si se ataca tras P2-1):

1. `LoRaConfig` en manifest.
2. `LlamaCppEngine.load(..., lora_paths=[...])` — soporta hasta N adapters apilados.
3. Modelfile acepta `ADAPTER <path-or-url>`.
4. `MESSAGE` en Modelfile se almacena y se prepende en `chat`.
5. Tests: LoRA real con phi-3-mini-4k + adapter HF.

**Archivos tocados:** 10 archivos, ~800 LOC, 20 tests.

---

### P3-3 — `REQUIRES` (version gating)

**Impacto:** Permite que un Modelfile diga "necesito HFL >= 0.4" y HFL rechace cargarlo si el binario es más viejo.
Requiere haber implementado Modelfile primero. 1 archivo, ~30 LOC, 3 tests.

---

## 4. Plan por fases

| Fase | Duración estimada | Alcance | Entregable |
|---|---|---|---|
| **Fase 1 — Paridad operacional** | 1-2 semanas | P0-2 (`/api/ps`), P0-3 (`/api/show`), P0-4 (`keep_alive`), P1-4 (`stop`), P1-5 (pull progress) | Release 0.4.0 — ecosystem Ollama ve HFL como drop-in para management |
| **Fase 2 — Embeddings** | 1 semana | P0-1 | Release 0.4.1 — RAG pipelines (LangChain, LlamaIndex) funcionan |
| **Fase 3 — Structured outputs** | 1-2 semanas | P0-5 | Release 0.4.2 — tool-calling avanzado + JSON mode OpenAI-compatible |
| **Fase 4 — Vision** | 2-3 semanas | P0-6 | Release 0.5.0 (breaking: `content` puede ser list) — soporte LLaVA/Gemma3-V/Qwen2-VL |
| **Fase 5 — Sys/think/copy/métricas** | 1 semana | P1-1, P1-2, P1-3 | Release 0.5.1 |
| **Fase 6 — Modelfile** | 3-4 semanas | P2-1 (parser + `/api/create` + `hfl create`), P2-2 (blobs), P2-3 (template/raw) | Release 0.6.0 — full Modelfile support |
| **Fase 7 — Polish** | 1 semana | P2-4, P3-3 | Release 0.6.1 |
| **Fase 8 — LoRA (opcional)** | 2 semanas | P3-2 | Release 0.7.0 |

**Total estimado:** 12-17 semanas para paridad completa (excluyendo `/api/push` que se descarta).

---

## 5. Rationale de priorización

- **P0 = bloquea ecosistema.** Sin embeddings, `/api/ps`, `/api/show`, `keep_alive` o structured outputs, los SDKs
  oficiales de Ollama (Python, JS) fallan silenciosamente o devuelven UNKNOWN. También bloquean integraciones con Open
  WebUI, LibreChat, Continue, y agentes basados en LangChain.
- **P1 = UX/operaciones.** Clientes avanzados lo piden, pero los básicos funcionan sin ello.
- **P2 = feature completion.** Modelfile es **la** abstracción icónica de Ollama; implementarla bien es un proyecto serio,
  no un fin de semana.
- **P3 = opcional / non-goal.** `/api/push` requiere ser un registry, cosa que HFL no quiere ser (filosofía local-first +
  integración con HF Hub existente).

---

## 6. Riesgos y dependencias

| Riesgo | Mitigación |
|---|---|
| GBNF grammars en `llama-cpp-python` cambian entre versiones | Pin exacto de la versión en `pyproject.toml` + test de humo que valide schema JSON simple |
| Visión en Transformers requiere `AutoModelForVision2Seq` + processor específico por familia | Empezar con Gemma 3 (ya soportado en HFL) y Qwen2-VL; dejar LLaVA para Fase 4b |
| Modelfile parsing bug-prone (formato ambiguo en `PARAMETER stop "..."` con escapes) | Parser formal con `lark` o `pyparsing`, no regex; corpus de test de 50+ Modelfiles reales de `ollama.com/library` |
| Body multipart de `/api/create` con blobs pre-subidos requiere streaming correcto bajo el body-limit | Whitelistar `/api/blobs/*` del body-limit; límite propio por blob (default 100 GB para GGUFs grandes) |
| `keep_alive=-1` (infinito) combinado con `queue_max_inflight=1` puede llenar RAM | Warning en logs + métrica `hfl_model_pool_pinned_total` |
| Structured outputs bloquea en modelos pequeños (salidas malformadas) | Fallback a regex post-hoc + warning log, no error |

---

## 7. Métricas de éxito

- `ollama-python` 0.6+ conecta a `http://localhost:11434` de HFL y pasa el 100% de su suite de ejemplos (`chat.py`,
  `generate.py`, `embed.py`, `ps.py`, `show.py`).
- LangChain `OllamaLLM` + `OllamaEmbeddings` funcionan sin patching.
- Open WebUI detecta el servidor, lista modelos con UNTIL y permite stop/start desde la UI.
- `curl http://localhost:11434/api/show -d '{"model":"qwen-coder:7b"}'` devuelve un JSON con las capabilities correctas.
- Structured outputs: pasar el suite de `instructor` Python con HFL como backend.
- Vision: Gemma 3 4B describe una imagen de gato con accuracy humano-equivalente.

---

## 8. Cambios no triviales al core

Implementar este plan requiere tocar:

- `src/hfl/engine/base.py` — ampliar `ChatMessage` (images), `GenerationConfig` (grammar, response_format, keep_alive),
  `GenerationResult` (4 timings).
- `src/hfl/models/manifest.py` — nuevos campos (`clip_model_path`, `parent_digest`, `blob_ref_count`, `modelfile_source`,
  `capabilities`, `lora_paths`).
- `src/hfl/api/schemas/` — todos los schemas tocados para los nuevos campos de request/response.
- `src/hfl/engine/model_pool.py` — `keep_alive_deadline`, `set_keep_alive()`, evicción por deadline.
- `src/hfl/converter/` — nuevo parser de Modelfile.
- `src/hfl/hub/blobs.py` — nuevo módulo completo.

Es **mucho** código nuevo pero cada fase es independiente y testeable. La filosofía que ha funcionado en las últimas
rondas sigue aplicando: una fase → un commit por ítem → ci-local entre ítems → release atómico al final de la fase.

---

## 9. Fuera de alcance explícito

- Replicar el cliente oficial de Ollama (`ollama` TS/Python SDK). HFL es servidor; sus clientes siguen siendo el SDK de
  OpenAI, Anthropic, etc. — la paridad es del servidor.
- `ollama launch <integration>`. Los wrappers de Claude Code / Copilot son marketing de Ollama, no features técnicas.
- Hospedar un registry propio. Usar HuggingFace Hub directamente mantiene la filosofía "local-first, no ataduras".
- Image generation experimental (`/api/generate` con `width/height/steps`). Es experimental hasta en Ollama.

---

## 10. Referencias

- Ollama API reference (Abril 2026): https://docs.ollama.com/api
- Ollama CLI reference: https://docs.ollama.com/cli
- Ollama Modelfile reference: https://docs.ollama.com/modelfile
- Ollama structured outputs: https://docs.ollama.com/capabilities/structured-outputs
- Ollama multimodal support: https://deepwiki.com/ollama/ollama/7.3-multimodal-and-vision-support
- HFL spec interna: `hfl-tool-calling-spec.md`, `guia-hf-local-deploy-hfl.md` (directorio padre)
- HFL estado actual: `CHANGELOG.md` → 0.3.5

---

*Documento generado como análisis arquitectónico. Las estimaciones de LOC y horas son aproximadas; cada fase debe
arrancar con su propio breakdown detallado y tests-first.*
