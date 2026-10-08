"""Self-registering adapter registry keyed by (stage, name)."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from csi_bench.core.errors import ConfigError

_REGISTRY: dict[tuple[str, str], type] = {}


def register[T: type](stage: str, name: str) -> Callable[[T], T]:
    """Register an adapter class. It must declare pydantic `Params` and int `VERSION`."""

    def decorator(cls: T) -> T:
        key = (stage, name)
        if key in _REGISTRY:
            raise ConfigError(f"adapter already registered: {stage}/{name}")
        params: Any = getattr(cls, "Params", None)
        if not (isinstance(params, type) and issubclass(params, BaseModel)):
            raise ConfigError(f"adapter {stage}/{name} must declare a pydantic Params")
        version: Any = getattr(cls, "VERSION", None)
        if not isinstance(version, int) or isinstance(version, bool):
            raise ConfigError(f"adapter {stage}/{name} must declare an int VERSION")
        _REGISTRY[key] = cls
        return cls

    return decorator


def get(stage: str, name: str) -> type:
    try:
        return _REGISTRY[(stage, name)]
    except KeyError:
        raise ConfigError(f"unknown adapter: {stage}/{name}") from None


def names(stage: str) -> list[str]:
    return sorted(n for s, n in _REGISTRY if s == stage)


def affecting_params(params: BaseModel) -> dict[str, Any]:
    """JSON params except those fields marked `affects_result=False` (AD-5)."""
    dumped = params.model_dump(mode="json")
    for name, field in type(params).model_fields.items():
        extra = field.json_schema_extra
        if isinstance(extra, dict) and extra.get("affects_result") is False:
            dumped.pop(name, None)
    return dumped


def reset() -> None:
    """Clear all registrations (for tests)."""
    _REGISTRY.clear()
