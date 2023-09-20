"""Definition of common methods."""
from typing import Any


class Registry(type):
    """Define a class for a registry."""

    _registry: dict[str, type[Any]] = {}

    def __new__(cls, name, bases, attrs):
        """Add the class to the registry"""
        new_cls = type.__new__(cls, name, bases, attrs)
        cls._registry[new_cls.__name__.lower()] = new_cls
        return new_cls

    @classmethod
    def get(cls, name: str) -> type | None:
        """Return a class if the name exists."""
        return cls._registry.get(name.lower(), None)

    @classmethod
    def get_all(cls):
        return cls._registry
