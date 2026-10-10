"""Fluent builder for `ExperimentConfig`; builds a plain dict and validates it once."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Self

from csi_bench.config.experiment import ExperimentConfig, config_from_dict
from csi_bench.core.errors import ConfigError


def _spec(entry: str | dict[str, Any] | tuple[str, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(entry, str):
        return {"name": entry}
    if isinstance(entry, tuple):
        if len(entry) != 2 or not isinstance(entry[1], Mapping):
            raise ConfigError(
                f"builder.features: tuple entry must be (name, params mapping), got {entry!r}"
            )
        name, params = entry
        return {"name": name, "params": dict(params)}
    return deepcopy(entry)


class ExperimentBuilder:
    def __init__(self) -> None:
        self._raw: dict[str, Any] = {}

    def dataset(self, name: str, split: str | None = None) -> Self:
        self._raw["dataset"] = name
        if split is not None:
            self._raw["split"] = split
        return self

    def rooms(self, *rooms: str) -> Self:
        self._raw["selection"] = {"rooms": list(rooms)}
        return self

    def preprocessing(self, *steps: Any) -> Self:
        """Preset names (`"ant2"`), `{name, params}` mappings or `{step: ...}` shorthand."""
        self._raw["preprocessing"] = [deepcopy(s) for s in steps]
        return self

    def window(self, length: int, stride: int = 1) -> Self:
        self._raw["window_len"] = length
        self._raw["stride"] = stride
        return self

    def features(self, *extractors: str | dict[str, Any] | tuple[str, dict[str, Any]]) -> Self:
        self._raw["extractors"] = [_spec(e) for e in extractors]
        return self

    def feature_selection(self, spec: Any) -> Self:
        """Shorthand (`{"mrmr": {"k": [2]}}`), `{name, params}` or `None`."""
        self._raw["feature_selection"] = deepcopy(spec)
        return self

    def head(self, name: str, **params: Any) -> Self:
        self._raw["head"] = {"name": name, "params": params} if params else {"name": name}
        return self

    def seeds(self, *seeds: int) -> Self:
        self._raw["seeds"] = list(seeds)
        return self

    def role(self, role: str) -> Self:
        self._raw["role"] = role
        return self

    def build(self) -> ExperimentConfig:
        return config_from_dict(deepcopy(self._raw), "ExperimentBuilder")
