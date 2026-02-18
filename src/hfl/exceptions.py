# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Jerarquía de excepciones personalizadas para hfl.

Usar excepciones específicas permite:
- Mejor manejo de errores en el código cliente
- Mensajes de error más claros para el usuario
- Facilita testing y debugging
"""


class HFLError(Exception):
    """Excepción base para todos los errores de hfl."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\n{self.details}"
        return self.message


# --- Errores de Modelos ---


class ModelNotFoundError(HFLError):
    """El modelo solicitado no se encontró en el registro local."""

    def __init__(self, model_name: str):
        super().__init__(
            f"Modelo no encontrado: {model_name}",
            "Usa 'hfl list' para ver modelos disponibles o 'hfl pull' para descargar.",
        )
        self.model_name = model_name


class ModelAlreadyExistsError(HFLError):
    """El modelo ya existe en el registro local."""

    def __init__(self, model_name: str):
        super().__init__(f"El modelo ya existe: {model_name}")
        self.model_name = model_name


# --- Errores de Descarga ---


class DownloadError(HFLError):
    """Error durante la descarga de un modelo."""

    def __init__(self, repo_id: str, reason: str):
        super().__init__(
            f"Error descargando {repo_id}",
            reason,
        )
        self.repo_id = repo_id


class NetworkError(DownloadError):
    """Error de red durante la descarga."""

    def __init__(self, repo_id: str, reason: str):
        super().__init__(repo_id, f"Error de red: {reason}")


# --- Errores de Conversión ---


class ConversionError(HFLError):
    """Error durante la conversión de formato del modelo."""

    def __init__(self, source_format: str, target_format: str, reason: str):
        super().__init__(
            f"Error convirtiendo de {source_format} a {target_format}",
            reason,
        )
        self.source_format = source_format
        self.target_format = target_format


class ToolNotFoundError(ConversionError):
    """Herramienta de conversión no encontrada."""

    def __init__(self, tool_name: str):
        super().__init__(
            "N/A",
            "N/A",
            f"Herramienta no encontrada: {tool_name}. "
            "Se instalará automáticamente en el próximo intento.",
        )
        self.tool_name = tool_name


# --- Errores de Licencia ---


class LicenseError(HFLError):
    """Error relacionado con licencias de modelos."""

    pass


class LicenseNotAcceptedError(LicenseError):
    """El usuario no aceptó la licencia del modelo."""

    def __init__(self, repo_id: str, license_type: str):
        super().__init__(
            f"Licencia no aceptada para {repo_id}",
            f"Este modelo requiere aceptar la licencia '{license_type}'.",
        )
        self.repo_id = repo_id
        self.license_type = license_type


class GatedModelError(LicenseError):
    """El modelo requiere aceptación previa en HuggingFace."""

    def __init__(self, repo_id: str):
        super().__init__(
            f"Modelo gated: {repo_id}",
            f"Debes aceptar los términos en https://huggingface.co/{repo_id} "
            "antes de poder descargarlo.",
        )
        self.repo_id = repo_id


# --- Errores de Motor de Inferencia ---


class EngineError(HFLError):
    """Error del motor de inferencia."""

    pass


class ModelNotLoadedError(EngineError):
    """Se intentó usar el motor sin cargar un modelo."""

    def __init__(self):
        super().__init__(
            "Modelo no cargado",
            "Debes cargar un modelo antes de generar texto.",
        )


class MissingDependencyError(EngineError):
    """Dependencia del motor no instalada."""

    def __init__(self, engine_name: str, package: str, install_cmd: str):
        super().__init__(
            f"Dependencia faltante para {engine_name}",
            f"Instala con: {install_cmd}",
        )
        self.engine_name = engine_name
        self.package = package
        self.install_cmd = install_cmd


class OutOfMemoryError(EngineError):
    """No hay suficiente memoria para el modelo."""

    def __init__(self, required_gb: float, available_gb: float):
        super().__init__(
            "Memoria insuficiente",
            f"El modelo requiere ~{required_gb:.1f}GB pero solo hay {available_gb:.1f}GB disponibles. "
            "Intenta con un modelo más pequeño o una cuantización más agresiva (Q4_K_S, Q3_K_M).",
        )
        self.required_gb = required_gb
        self.available_gb = available_gb


# --- Errores de Autenticación ---


class AuthenticationError(HFLError):
    """Error de autenticación con HuggingFace."""

    pass


class InvalidTokenError(AuthenticationError):
    """Token de HuggingFace inválido."""

    def __init__(self):
        super().__init__(
            "Token inválido",
            "Verifica tu token en https://huggingface.co/settings/tokens "
            "o usa 'hfl login' para configurarlo.",
        )


class TokenRequiredError(AuthenticationError):
    """Se requiere token para acceder al recurso."""

    def __init__(self, repo_id: str):
        super().__init__(
            f"Token requerido para {repo_id}",
            "Usa 'hfl login' para configurar tu token de HuggingFace.",
        )
        self.repo_id = repo_id


# --- Errores de Configuración ---


class ConfigurationError(HFLError):
    """Error de configuración."""

    pass


class InvalidConfigError(ConfigurationError):
    """Valor de configuración inválido."""

    def __init__(self, key: str, value: str, valid_values: list[str] | None = None):
        details = f"Valor inválido para '{key}': {value}"
        if valid_values:
            details += f"\nValores válidos: {', '.join(valid_values)}"
        super().__init__("Error de configuración", details)
        self.key = key
        self.value = value
