# hfl

[![Licencia: HRUL v1.0](https://img.shields.io/badge/Licencia-HRUL%20v1.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/ggalancs/hfl/actions/workflows/ci.yml/badge.svg)](https://github.com/ggalancs/hfl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ggalancs/hfl/branch/main/graph/badge.svg)](https://codecov.io/gh/ggalancs/hfl)

Ejecuta modelos de HuggingFace localmente como Ollama.

> **[English version](README.md)**

## Por qué HFL?

**Ollama tiene un catálogo curado de ~500 modelos. HuggingFace Hub tiene más de 500,000.**

Si quieres ejecutar un modelo que no está en el catálogo de Ollama — un fine-tune específico, un lanzamiento reciente de un laboratorio pequeño, un modelo de nicho — tienes que descargar manualmente desde HuggingFace, convertir a GGUF con llama.cpp, cuantizar y configurar la inferencia. **HFL automatiza todo esto en un solo comando.**

| Característica | Ollama | HFL |
|----------------|--------|-----|
| Catálogo de modelos | ~500 curados | 500K+ (todo HF Hub) |
| Auto-conversión | No necesaria (pre-convertidos) | Sí (safetensors→GGUF) |
| Facilidad de uso | Excelente | Buena |
| Compatible con API OpenAI | Sí | Sí |
| Compatible con API Ollama | Nativo | Sí (drop-in) |
| API Anthropic Messages | No | Sí (compatible con Claude Code) |
| Tool calling estructurado | Sí | Sí (qwen / llama3 / mistral) |
| Multi-backend | solo llama.cpp | llama.cpp + Transformers + vLLM |
| Verificación de licencia | No | Sí (5 niveles de riesgo) |
| Trazabilidad legal | No | Sí (log de procedencia) |
| Madurez | Alta (establecido) | Alpha (v0.12.0) |

**HFL no compite con Ollama — lo complementa.** Usa Ollama para modelos curados; usa HFL cuando necesites algo del ecosistema completo de HuggingFace.

## Características

- **CLI y API**: Interfaz CLI completa más API REST compatible con OpenAI, Ollama y Anthropic
- **Búsqueda de Modelos**: Búsqueda paginada interactiva en HuggingFace Hub (como `more`)
- **Múltiples Backends**: llama.cpp (GGUF/CPU), Transformers (GPU nativo), vLLM (producción)
- **Conversión Automática**: Descarga modelos de HuggingFace y convierte a GGUF automáticamente
- **Cuantización Inteligente**: Soporta niveles de cuantización Q2_K hasta F16
- **Texto a Voz**: Soporte nativo TTS con Bark, SpeechT5, Coqui XTTS y más
- **Tool calling estructurado**: Protocolo `tools` / `tool_calls` compatible con Ollama, con parsers por familia para qwen, llama3 y mistral — los agentes funcionan sin configuración extra
- **Cola de inferencia acotada**: serialización de peticiones en el servidor con backpressure explícito (429 / 503), cabeceras `X-Queue-Depth` en vivo y `GET /healthz` para orquestadores
- **Compatible Drop-in**: Funciona como reemplazo de Ollama con herramientas existentes
- **Internacionalizado**: Soporte completo i18n (Inglés, Español) - configura `HFL_LANG` para cambiar idioma

## Cómo Funciona

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Arquitectura de HFL                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐         ┌──────────────────┐         ┌─────────────────┐   │
│  │  hfl pull   │───────▶ │  HuggingFace Hub │───────▶ │  Almacenamiento │   │
│  │             │         │                  │         │   ~/.hfl/       │   │
│  └─────────────┘         │  • API Búsqueda  │         │   ├── models/   │   │
│        │                 │  • Descarga      │         │   ├── cache/    │   │
│        │                 │  • Info licencia │         │   └── registry  │   │
│        ▼                 └──────────────────┘         └─────────────────┘   │
│  ┌─────────────┐                                              │             │
│  │  Conversor  │◀─────────────────────────────────────────────┘             │
│  │             │                                                            │
│  │ safetensors │──────────▶ GGUF (cuantizado Q2_K...F16)                    │
│  └─────────────┘                                                            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐         ┌──────────────────┐         ┌─────────────────┐   │
│  │  hfl run    │───────▶ │ Motor Inferencia │───────▶ │      Chat       │   │
│  │             │         │                  │         │   Interactivo   │   │
│  └─────────────┘         │  • llama.cpp     │         └─────────────────┘   │
│                          │  • Transformers  │                               │
│                          │  • vLLM          │                               │
│                          └──────────────────┘                               │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐         ┌──────────────────┐         ┌─────────────────┐   │
│  │  hfl serve  │───────▶ │   API REST       │───────▶ │  OpenAI SDK /   │   │
│  │             │         │                  │         │ clientes Ollama │   │
│  └─────────────┘         │  • /v1/chat/...  │         └─────────────────┘   │
│                          │  • /api/chat     │                               │
│                          │  • /api/generate │                               │
│                          └──────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Resumen del Flujo:**
1. **Pull**: Descargar de HuggingFace Hub → Convertir a GGUF (si es necesario) → Almacenar localmente
2. **Run**: Cargar modelo en motor de inferencia → Iniciar sesión de chat interactivo
3. **Serve**: Iniciar servidor API → Aceptar peticiones compatibles con OpenAI/Ollama

## Prerrequisitos

- **Python 3.10+** (requerido)
- **git** (para clonar llama.cpp durante la primera conversión)
- **cmake** y **compilador C++** (para compilar herramientas de cuantización de llama.cpp)
  - macOS: `xcode-select --install`
  - Ubuntu/Debian: `sudo apt install build-essential cmake`
  - Windows: Instalar Visual Studio Build Tools

> **Nota:** Las herramientas de compilación solo son necesarias si conviertes modelos safetensors a GGUF. Si solo usas modelos GGUF pre-cuantizados, no son necesarias.

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/ggalancs/hfl
cd hfl

# Instalación básica (CPU + GGUF)
pip install .

# Con soporte GPU (Transformers + bitsandbytes)
pip install ".[transformers]"

# Con soporte TTS (Bark, SpeechT5)
pip install ".[tts]"

# Con Coqui TTS (XTTS-v2, VITS)
pip install ".[coqui]"

# Con vLLM para producción
pip install ".[vllm]"

# Todo incluido
pip install ".[all]"
```

## Inicio Rápido

### Descargar un Modelo

```bash
# Descargar con cuantización Q4_K_M por defecto
hfl pull meta-llama/Llama-3.3-70B-Instruct

# Especificar nivel de cuantización
hfl pull meta-llama/Llama-3.3-70B-Instruct --quantize Q5_K_M

# Mantener como safetensors (para inferencia GPU)
hfl pull mistralai/Mistral-7B-Instruct-v0.3 --format safetensors

# Descargar con un alias personalizado para referencia más fácil
hfl pull meta-llama/Llama-3.3-70B-Instruct --alias llama70b
```

### Chat Interactivo

```bash
# Iniciar chat con un modelo
hfl run llama-3.3-70b-instruct-q4_k_m

# Con prompt de sistema
hfl run llama-3.3-70b-instruct-q4_k_m --system "Eres un experto en Python"
```

### Servidor API

```bash
# Iniciar servidor (puerto 11434 por defecto, igual que Ollama)
hfl serve

# Pre-cargar un modelo
hfl serve --model llama-3.3-70b-instruct-q4_k_m

# Host/puerto personalizado
hfl serve --host 0.0.0.0 --port 8080
```

### Texto a Voz (TTS)

HFL soporta modelos TTS de HuggingFace como Bark, SpeechT5 y Coqui XTTS.

```bash
# Descargar un modelo TTS (no necesita conversión GGUF)
hfl pull suno/bark-small --alias bark

# Sintetizar texto a archivo de audio
hfl tts bark "Hola, esto es una prueba." -o salida.wav

# Sintetizar y reproducir directamente (requiere sounddevice)
hfl speak bark "Hola, esto es una prueba."

# Con opciones
hfl tts bark "Hello world" --lang en --output english.wav --speed 0.9
hfl speak bark "Habla rápida" --speed 1.5
```

**Opciones TTS:**
- `--output, -o`: Ruta del archivo de salida (por defecto: output.wav)
- `--lang, -l`: Código de idioma (en, es, fr, etc.)
- `--voice, -v`: Voz/hablante a usar
- `--speed, -s`: Multiplicador de velocidad (0.25-4.0)
- `--rate, -r`: Frecuencia de muestreo en Hz
- `--format, -f`: Formato de audio (wav, mp3, ogg)

**API TTS:**
```bash
# Endpoint compatible con OpenAI
curl -X POST http://localhost:11434/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "bark", "input": "Hola mundo", "voice": "alloy"}' \
  --output speech.wav

# Endpoint nativo HFL
curl -X POST http://localhost:11434/api/tts \
  -H "Content-Type: application/json" \
  -d '{"model": "bark", "text": "Hola mundo", "language": "es"}' \
  --output speech.wav
```

### Buscar Modelos en HuggingFace

```bash
# Buscar modelos (paginado como 'more')
hfl search llama

# Buscar solo modelos con archivos GGUF
hfl search mistral --gguf

# Personalizar paginación y resultados
hfl search phi --limit 50 --page-size 5

# Ordenar por likes en vez de descargas
hfl search qwen --sort likes
```

**Controles de navegación:**
- `ESPACIO` / `ENTER` - Página siguiente
- `p` - Página anterior
- `q` / `ESC` - Salir

### Gestión de Modelos

```bash
# Listar todos los modelos locales
hfl list

# Mostrar detalles del modelo
hfl inspect llama-3.3-70b-instruct-q4_k_m

# Eliminar un modelo
hfl rm llama-3.3-70b-instruct-q4_k_m

# Establecer un alias para un modelo existente
hfl alias llama-3.3-70b-instruct-q4_k_m llama70b

# Ahora usa el alias en cualquier comando
hfl run llama70b
hfl inspect llama70b
```

## Endpoints de API

### Compatible con OpenAI

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Hola!"}]
  }'
```

### Compatible con Ollama

```bash
curl http://localhost:11434/api/chat \
  -d '{
    "model": "llama-3.3-70b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Hola!"}]
  }'
```

### Usando OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="no-necesario"
)

response = client.chat.completions.create(
    model="llama-3.3-70b-instruct-q4_k_m",
    messages=[{"role": "user", "content": "Explica la computación cuántica"}],
)
print(response.choices[0].message.content)
```

## Tool Calling (Agentes)

HFL implementa el protocolo de **tool calling estructurado** compatible
con Ollama, de forma que los agentes escritos con el SDK de Ollama
pueden ejecutar bucles multi-turno de herramientas directamente. Cuando
el cliente envía `tools` en `/api/chat`, HFL los propaga a través de la
plantilla nativa del modelo (qwen3 `<tool_call>`, llama3
`<|python_tag|>`, mistral `[TOOL_CALLS]`), parsea la respuesta a
`message.tool_calls` canónicos con `arguments` como objeto JSON, y
acepta resultados `role: "tool"` en el turno siguiente.

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b-q4_k_m",
    "stream": false,
    "messages": [
      {"role":"system","content":"Debes llamar a write_wiki, nunca respondas con texto."},
      {"role":"user","content":"Guarda Hola en topics/hello.md"}
    ],
    "tools":[{
      "type":"function",
      "function":{
        "name":"write_wiki",
        "description":"Crea o sobrescribe un artículo del wiki",
        "parameters":{
          "type":"object",
          "properties":{"path":{"type":"string"},"content":{"type":"string"}},
          "required":["path","content"]
        }
      }
    }]
  }'
```

Respuesta:

```json
{
  "model": "qwen3-32b-q4_k_m",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "write_wiki",
          "arguments": {"path": "topics/hello.md", "content": "Hola"}
        }
      }
    ]
  },
  "done": true
}
```

El parser por familia también gestiona un fallback genérico
`{"tool_call": {...}}` para plantillas que no se aplicaron
correctamente. En streaming (`stream: true`), HFL acumula toda la
respuesta y emite los `tool_calls` en el chunk final `done: true`. La
especificación completa está en
[`hfl-tool-calling-spec.md`](../hfl-tool-calling-spec.md) y la suite de
aceptación en `tests/test_tool_calling_acceptance.py`.

## Concurrencia y Backpressure

Los backends locales de inferencia (llama.cpp, transformers-GPU)
comparten una única instancia de modelo no reentrante. HFL los protege
con un **dispatcher de inferencia interno** que serializa las peticiones
sobre el motor cargado con una cola de espera acotada:

| Parámetro | Variable de entorno | Por defecto | Significado |
|---|---|---|---|
| Máx. en vuelo | `HFL_QUEUE_MAX_INFLIGHT` | `1` | Peticiones paralelas permitidas |
| Tamaño de cola | `HFL_QUEUE_MAX_SIZE` | `16` | Peticiones que pueden esperar |
| Timeout de adquisición | `HFL_QUEUE_ACQUIRE_TIMEOUT` | `60` | Segundos que puede esperar una petición |
| Habilitado | `HFL_QUEUE_ENABLED` | `true` | Interruptor global |

Cuando la cola de espera está llena, HFL responde **429** con un
envelope estructurado y `Retry-After`:

```json
{
  "error": "Inference queue is full",
  "code": "QUEUE_FULL",
  "category": "rate_limit",
  "retryable": true,
  "details": {"retry_after_seconds": 60, "queue_depth": 1, "max_queued": 1}
}
```

Si un llamante lleva más de `HFL_QUEUE_ACQUIRE_TIMEOUT` en cola, HFL
devuelve **503** con `code=QUEUE_TIMEOUT`. Cada respuesta incluye
`X-Queue-Depth`, `X-Queue-In-Flight`, `X-Queue-Max-Inflight` y
`X-Queue-Max-Size` para que los agentes apliquen backoff
proporcional. El estado en vivo también está disponible vía:

```bash
curl http://localhost:11434/healthz
# { "status":"ok", "models_loaded":[...], "queue_depth":0,
#   "queue_in_flight":0, "uptime_seconds":12345 }
```

Las tres superficies de API (Ollama, OpenAI, Anthropic) comparten el
mismo dispatcher, así que una llamada lenta en `/api/chat` bloquea
correctamente a `/v1/chat/completions` y `/v1/messages`.

## Niveles de Cuantización

| Nivel | Bits/peso | Calidad | Caso de Uso |
|-------|-----------|---------|-------------|
| Q2_K | ~2.5 | ~80% | Compresión extrema |
| Q3_K_M | ~3.5 | ~87% | RAM baja |
| **Q4_K_M** | ~4.5 | ~92% | **Por defecto - mejor balance** |
| Q5_K_M | ~5.0 | ~96% | Alta calidad |
| Q6_K | ~6.5 | ~97% | Premium |
| Q8_0 | ~8.0 | ~98%+ | Máxima calidad cuantizada |
| F16 | 16.0 | 100% | Sin cuantización |

## Requisitos de RAM

```
RAM necesaria ≈ (parámetros × bits_por_peso) / 8 + 2GB overhead

Ejemplo: Llama 3.3 70B con Q4_K_M
= (70B × 4.5) / 8 + 2GB ≈ 41.4 GB
```

| Tamaño Modelo | RAM Q4_K_M | Hardware Recomendado |
|---------------|------------|----------------------|
| 7B | ~5 GB | 8 GB RAM |
| 13B | ~9 GB | 16 GB RAM |
| 30B | ~20 GB | 32 GB RAM |
| 70B | ~42 GB | 48 GB+ RAM o GPU |

## Autenticación

Configura tu token de HuggingFace para descargas más rápidas y acceso a modelos restringidos:

```bash
# Login interactivo (recomendado - almacena token de forma segura)
hfl login

# O usa variable de entorno (más privado - no persistido)
export HF_TOKEN=hf_tu_token_aqui
```

Obtén tu token en: https://huggingface.co/settings/tokens

## Configuración

Variables de entorno:
- `HFL_HOME`: Directorio de datos (por defecto: `~/.hfl`)
- `HF_TOKEN`: Token de HuggingFace para modelos restringidos (alternativa a `hfl login`)
- `HFL_LANG`: Idioma de la interfaz (`en` para Inglés, `es` para Español). Por defecto Inglés.

### Soporte de Idiomas

hfl soporta múltiples idiomas. Configura la variable de entorno `HFL_LANG` para cambiar el idioma del CLI:

```bash
# Usar Español
export HFL_LANG=es
hfl --help

# Usar Inglés (por defecto)
export HFL_LANG=en
hfl --help
```

Idiomas soportados: Inglés (`en`), Español (`es`)

## Limitaciones Conocidas

Esta es una versión alpha v0.3.x. Las limitaciones conocidas incluyen:

- **Backend vLLM es experimental**: Implementación básica sin soporte completo de streaming
- **CORS es restrictivo por defecto**: mismo-origen solamente; habilitar con `cors_allow_all` o `cors_origins` explícito
- **Soporte Windows**: No completamente probado; se recomiendan sistemas Unix-like

### Autenticación de API

El servidor API soporta autenticación opcional mediante el flag `--api-key`:

```bash
# Iniciar servidor con autenticación
hfl serve --api-key tu-clave-secreta

# Las peticiones del cliente deben incluir la clave
curl -H "Authorization: Bearer tu-clave-secreta" http://localhost:11434/v1/models
# O
curl -H "X-API-Key: tu-clave-secreta" http://localhost:11434/v1/models
```

## Documentación

Documentación completa de arquitectura con diagramas disponible:

- **[📖 Ver Documentación de Arquitectura](https://htmlpreview.github.io/?https://github.com/ggalancs/hfl/blob/main/docs/hfl-arquitectura-completa.html)** - Documentación HTML interactiva con diagramas de arquitectura, descripciones de módulos y diagramas de flujo

La documentación cubre:
- Arquitectura del sistema y patrones de diseño
- Estructura de módulos y dependencias
- Lógica de selección de motor de inferencia
- Pipeline de conversión GGUF
- Características de cumplimiento legal
- Referencia de endpoints de API

> **Nota:** La documentación también está disponible en [Inglés](https://htmlpreview.github.io/?https://github.com/ggalancs/hfl/blob/main/docs/hfl-architecture-complete.html).

## Desarrollo

```bash
# Clonar e instalar en modo desarrollo
git clone https://github.com/ggalancs/hfl
cd hfl
pip install -e ".[dev]"

# Ejecutar tests
pytest

# Ejecutar tests con cobertura
pytest --cov=hfl --cov-report=term-missing

# Formatear código
ruff format .
ruff check . --fix
```

## Avisos Legales

### Cumplimiento de Exportación

hfl solo descarga modelos de pesos abiertos disponibles públicamente desde HuggingFace Hub. Los usuarios son responsables del cumplimiento de las regulaciones de control de exportación aplicables en su jurisdicción.

hfl no facilita el acceso a pesos de modelos cerrados o controlados para exportación.

### Licencias de Modelos

Los modelos descargados a través de hfl pueden tener sus propias restricciones de licencia. hfl muestra información de licencia antes de la descarga y la almacena con los metadatos del modelo. Los usuarios son responsables de cumplir con las licencias de los modelos.

Las restricciones comunes incluyen:
- **Solo uso no comercial** (CC-BY-NC, MRL)
- **Atribución requerida** (Llama, Gemma)
- **Restricciones de uso** (OpenRAIL)

Usa `hfl inspect <modelo>` para ver detalles de licencia de modelos descargados.

### Descargo de Responsabilidad

Los modelos de IA pueden generar contenido inexacto, sesgado o inapropiado. Los usuarios son los únicos responsables de evaluar y usar las salidas del modelo apropiadamente. Ver [DISCLAIMER.md](DISCLAIMER.md) para detalles completos.

## Marcas Registradas

"OpenAI" es una marca registrada de OpenAI, Inc. "Ollama" es una marca registrada de Ollama, Inc. "Hugging Face" y el logo de Hugging Face son marcas registradas de Hugging Face, Inc. Estas marcas se usan aquí solo con propósitos de identificación.

**hfl es un proyecto independiente y no está afiliado, respaldado, ni conectado oficialmente con Hugging Face, Inc., OpenAI, Inc., u Ollama, Inc.** Las referencias a estos servicios describen solo interoperabilidad técnica.

## Licencia

hfl se distribuye como source-available bajo la **hfl Responsible Use License (HRUL) v1.0**.

Esta licencia permite uso libre, modificación y distribución comercial con una condición: los trabajos derivados que se distribuyan públicamente deben mantener las características de cumplimiento legal (verificación de licencias, descargos de IA, seguimiento de procedencia, protecciones de privacidad y respeto de restricciones).

Eres libre de reescribir, extender, renombrar y vender derivados — simplemente no puedes eliminar las características de seguridad.

**Nota:** La HRUL no es una licencia open-source aprobada por la OSI. Es una licencia source-available con requisitos de uso responsable, inspirada en Apache 2.0, el copyleft de GPL, y la familia de licencias RAIL para IA.

Ver [LICENSE](LICENSE) para el texto completo y [LICENSE-FAQ.md](LICENSE-FAQ.md) para preguntas frecuentes.
