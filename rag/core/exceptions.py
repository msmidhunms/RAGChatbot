"""Exception hierarchy shared by all RAG modules."""


class RAGError(Exception):
    """Base class for every error raised by the rag package."""


class ConfigError(RAGError):
    """Configuration could not be loaded, parsed or validated."""


class RegistryError(RAGError):
    """A component name is unknown or registered twice."""


class MissingDependencyError(RAGError, ImportError):
    """An optional dependency needed by the selected component is not installed."""


class MissingCredentialsError(RAGError):
    """An API key or connection string required by the selected provider is not set."""
