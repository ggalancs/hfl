# hfl

[![Licencia: HRUL v1.0](https://img.shields.io/badge/Licencia-HRUL%20v1.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/ggalancs/hfl/actions/workflows/ci.yml/badge.svg)](https://github.com/ggalancs/hfl/actions/workflows/ci.yml)
[![Cobertura](https://img.shields.io/badge/cobertura-81%25-brightgreen.svg)](https://github.com/ggalancs/hfl)

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
| Multi-backend | solo llama.cpp | llama.cpp + Transformers + vLLM |
| Verificación de licencia | No | Sí (5 niveles de riesgo) |
| Trazabilidad legal | No | Sí (log de procedencia) |
| Madurez | Alta (establecido) | Pre-alpha (v0.1.0) |

**HFL no compite con Ollama — lo complementa.** Usa Ollama para modelos curados; usa HFL cuando necesites algo del ecosistema completo de HuggingFace.

## Características

- **CLI y API**: Interfaz CLI completa más API REST compatible con OpenAI y Ollama
- **Búsqueda de Modelos**: Búsqueda paginada interactiva en HuggingFace Hub (como `more`)
- **Múltiples Backends**: llama.cpp (GGUF/CPU), Transformers (GPU nativo), vLLM (producción)
- **Conversión Automática**: Descarga modelos de HuggingFace y convierte a GGUF automáticamente
- **Cuantización Inteligente**: Soporta niveles de cuantización Q2_K hasta F16
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
# Instalación básica (CPU + GGUF)
pip install hfl

# Con soporte GPU (Transformers + bitsandbytes)
pip install "hfl[transformers]"

# Con vLLM para producción
pip install "hfl[vllm]"

# Todo incluido
pip install "hfl[all]"
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

Esta es una versión v0.1.0. Las limitaciones conocidas incluyen:

- **Backend vLLM es experimental**: Implementación básica sin soporte completo de streaming
- **Sin limitación de tasa en API**: Solo las llamadas a HuggingFace Hub tienen límite de tasa
- **CORS es permisivo**: La API permite todos los orígenes (configurable en futuras versiones)
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
