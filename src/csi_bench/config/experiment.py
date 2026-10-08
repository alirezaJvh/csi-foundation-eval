"""The serializable `ExperimentConfig`, its YAML loader and `run_id` (AD-5).

The config is the single construction path: it is validated (types, registry names
and adapter params) before any dataset is loaded. This module imports only `core`.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from csi_bench.core import registry
from csi_bench.core.errors import ConfigError
from csi_bench.core.hashing import short_hash

STAGES = ("extractor", "head")


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rooms: list[str]


class AdapterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    split: str | None = None
    selection: Selection
    window_len: int
    stride: int = 1
    extractors: list[AdapterSpec]
    head: AdapterSpec
    seeds: list[int] = Field(default_factory=lambda: [0])
    role: Literal["comparison", "reference"]


def _named_error(error: ValidationError) -> str:
    first = error.errors()[0]
    loc = ".".join(str(part) for part in first["loc"])
    return f"{loc}: {first['msg']}" if loc else first["msg"]


def _validate_params(cls: Any, params: Mapping[str, Any], path: str) -> None:
    try:
        cls.Params(**params)
    except ValidationError as e:
        raise ConfigError(f"{path}: {_named_error(e)}") from e


def _check_adapter(stage: str, spec: AdapterSpec, path: str) -> Any:
    try:
        cls = registry.get(stage, spec.name)
    except ConfigError:
        known = ", ".join(registry.names(stage)) or "none"
        raise ConfigError(
            f"{path}.name: unknown {stage} {spec.name!r}; registered: {known}"
        ) from None
    _validate_params(cls, spec.params, f"{path}.params")
    return cls


def validate_registry(config: ExperimentConfig) -> None:
    """Check adapter names and their params against the (already discovered) registry."""
    for i, spec in enumerate(config.extractors):
        _check_adapter("extractor", spec, f"extractors.{i}")
    _check_adapter("head", config.head, "head")


def load_config(path: str | Path) -> ExperimentConfig:
    """Read YAML into a validated `ExperimentConfig`, or raise `ConfigError`."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"{path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: config must be a mapping")
    try:
        config = ExperimentConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"{path}: {_named_error(e)}") from e
    validate_registry(config)
    return config


def resolved_body(config: ExperimentConfig) -> dict[str, Any]:
    """Canonical config body for the run id: no seeds, adapter versions, result params only."""
    body = config.model_dump(mode="json")
    body.pop("seeds", None)
    extractors = []
    for i, spec in enumerate(config.extractors):
        cls = _check_adapter("extractor", spec, f"extractors.{i}")
        extractors.append(
            {
                "name": spec.name,
                "version": cls.VERSION,
                "params": registry.affecting_params(cls.Params(**spec.params)),
            }
        )
    body["extractors"] = extractors
    cls = _check_adapter("head", config.head, "head")
    body["head"] = {
        "name": config.head.name,
        "version": cls.VERSION,
        "params": registry.affecting_params(cls.Params(**config.head.params)),
    }
    return body


def compute_run_id(config: ExperimentConfig) -> str:
    """Short hash of the resolved config plus adapter `VERSION`s, excluding `seeds` (AD-5)."""
    return short_hash(resolved_body(config))


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Write the resolved config as YAML so a rerun reproduces the same `run_id`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    return path
