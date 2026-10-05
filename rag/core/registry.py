"""Name -> factory registries that make every pipeline component pluggable.

Each component family (loaders, splitters, stores, rerankers, ...) owns one
``Registry``. A backend is added by decorating a factory function::

    SPLITTERS = Registry("splitter")

    @SPLITTERS.register("recursive")
    def build_recursive(cfg): ...

and selected at runtime by the name given in the config. Factories should
import heavy or optional libraries inside their body (see ``require``) so
that only the backends actually chosen need to be installed.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Generic, TypeVar

from rag.core.exceptions import MissingDependencyError, RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, *, override: bool = False) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            self.add(name, obj, override=override)
            return obj

        return decorator

    def add(self, name: str, obj: T, *, override: bool = False) -> None:
        key = name.lower()
        if key in self._items and not override:
            raise RegistryError(f"{self.kind} '{key}' is already registered")
        self._items[key] = obj

    def get(self, name: str) -> T:
        try:
            return self._items[name.lower()]
        except KeyError:
            available = ", ".join(self.available()) or "none"
            raise RegistryError(f"unknown {self.kind} '{name}'. Available: {available}") from None

    def available(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.available())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.available()})"


def require(module: str, extra: str | None = None) -> ModuleType:
    """Import an optional dependency or explain how to install it.

    ``extra`` names the pyproject extra that provides ``module``; ``None``
    means it is a core dependency.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        fix = f"pip install 'ragchatbot[{extra}]'" if extra else "pip install -r requirements.txt"
        raise MissingDependencyError(
            f"'{module}' is required for this component. Install it with: {fix}"
        ) from exc
