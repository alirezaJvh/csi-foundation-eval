"""The serializable `ExperimentConfig`, its YAML loader and `run_id` (AD-5).

The config is the single construction path: it is validated (types, registry names
and adapter params) before any dataset is loaded. This module imports only `core`.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from csi_bench.config.presets import PREPROCESS_PRESETS
from csi_bench.core import registry
from csi_bench.core.errors import ConfigError, DataError
from csi_bench.core.hashing import sha256_hex, short_hash
from csi_bench.core.splitplan import (
    check_split_role,
    check_split_rooms,
    find_split,
    read_plan_params,
    read_plan_splitter,
    read_plan_unit,
)


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rooms: list[str] = Field(min_length=1)


class AdapterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


def _expand_preprocessing(raw: Any) -> Any:
    """Normalize shorthand and preset entries to explicit `{name, params}` mappings."""
    if not isinstance(raw, list):
        return raw
    out: list[Any] = []
    for i, entry in enumerate(raw):
        where = f"preprocessing.{i}"
        if isinstance(entry, str):
            if entry not in PREPROCESS_PRESETS:
                known = ", ".join(sorted(PREPROCESS_PRESETS))
                raise ValueError(f"{where}: unknown preset {entry!r}; presets: {known}")
            for step in PREPROCESS_PRESETS[entry]:
                out.append({"name": step["name"], "params": deepcopy(step["params"])})
        elif isinstance(entry, dict) and "name" in entry:
            out.append(entry)
        elif isinstance(entry, dict) and len(entry) == 1:
            ((name, value),) = entry.items()
            if isinstance(value, list):
                params: Any = {"ids": value}
            elif value is None:
                params = {}
            elif isinstance(value, dict):
                params = value
            else:
                raise ValueError(f"{where}: params of {name!r} must be a list or mapping")
            out.append({"name": name, "params": params})
        else:
            raise ValueError(f"{where}: expected a preset name, {{name, params}} or {{step: ...}}")
    return out


def _expand_feature_selection(raw: Any) -> Any:
    """Normalize `{manual: [...]}` / `{mrmr: {...}}` to `{name, params}`; keep explicit form."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("feature_selection: expected a mapping or null")
    if "name" in raw:
        return raw
    if len(raw) != 1 or next(iter(raw)) not in ("manual", "mrmr"):
        raise ValueError(
            "feature_selection: expected exactly one of 'manual' or 'mrmr' "
            f"(or {{name, params}}), got {sorted(map(str, raw))}"
        )
    ((name, value),) = raw.items()
    if name == "manual":
        if not isinstance(value, list):
            raise ValueError("feature_selection.manual: expected a list of columns or families")
        return {"name": "manual", "params": {"columns": value}}
    if not isinstance(value, dict):
        raise ValueError("feature_selection.mrmr: expected a mapping like {k: [2, 3]}")
    return {"name": "mrmr", "params": value}


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    split: str | None = None
    selection: Selection
    preprocessing: list[AdapterSpec] = Field(default_factory=list)
    window_len: int = Field(ge=1)
    stride: int = Field(default=1, ge=1)
    extractors: list[AdapterSpec] = Field(min_length=1)
    feature_selection: AdapterSpec | None = None
    head: AdapterSpec
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    role: Literal["comparison", "reference"]

    @field_validator("preprocessing", mode="before")
    @classmethod
    def _expand(cls, raw: Any) -> Any:
        return _expand_preprocessing(raw)

    @field_validator("feature_selection", mode="before")
    @classmethod
    def _expand_selector(cls, raw: Any) -> Any:
        return _expand_feature_selection(raw)

    @field_validator("extractors")
    @classmethod
    def _unique_extractors(cls, specs: list[AdapterSpec]) -> list[AdapterSpec]:
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError("extractor names must be unique")
        return specs

    @field_validator("seeds")
    @classmethod
    def _unique_seeds(cls, seeds: list[int]) -> list[int]:
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        return seeds


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
    _check_adapter("dataset", AdapterSpec(name=config.dataset), "dataset")
    # no usable plan yet: `run` checks again once it loads one
    try:
        plan_path = find_split(config.dataset, config.split)
        unit = read_plan_unit(plan_path)
    except DataError:
        plan_path, unit = None, "block"
    check_split_role(unit, config.role)
    if plan_path is not None:
        try:
            splitter, params = read_plan_splitter(plan_path), read_plan_params(plan_path)
        except DataError:
            splitter, params = None, {}
        if splitter is not None:
            check_split_rooms(splitter, params, config.selection.rooms, config.role)
    for i, spec in enumerate(config.preprocessing):
        _check_adapter("preprocess", spec, f"preprocessing.{i}")
        if spec.params.get("fit_scope") == "whole_recording" and config.role != "reference":
            raise ConfigError(
                f"preprocessing.{i}.params.fit_scope: whole_recording requires role: reference"
            )
    for i, spec in enumerate(config.extractors):
        cls = _check_adapter("extractor", spec, f"extractors.{i}")
        if config.window_len < cls.min_window_len:
            raise ConfigError(
                f"extractors.{i}.name: window_len {config.window_len} "
                f"< min_window_len {cls.min_window_len} for {spec.name}"
            )
    if config.feature_selection is not None:
        _check_adapter("selector", config.feature_selection, "feature_selection")
    _check_adapter("head", config.head, "head")


def config_from_dict(raw: Any, source: str = "config") -> ExperimentConfig:
    """The single construction path: dict -> `model_validate` -> `validate_registry`."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: config must be a mapping")
    try:
        config = ExperimentConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"{source}: {_named_error(e)}") from e
    validate_registry(config)
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Read YAML into a validated `ExperimentConfig`, or raise `ConfigError`."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"{path}: {e}") from e
    return config_from_dict(raw, str(path))


def resolved_body(config: ExperimentConfig, split_root: str | Path = "splits") -> dict[str, Any]:
    """Canonical config body for the run id: no seeds, adapter versions, result params only.

    `split` is resolved to the committed plan's hash, so replacing the plan changes the id.
    """
    body = config.model_dump(mode="json")
    body.pop("seeds", None)
    body["split"] = find_split(config.dataset, config.split, split_root).stem
    body["dataset_version"] = _check_adapter(
        "dataset", AdapterSpec(name=config.dataset), "dataset"
    ).VERSION
    steps = []
    for i, spec in enumerate(config.preprocessing):
        cls = _check_adapter("preprocess", spec, f"preprocessing.{i}")
        steps.append(
            {
                "name": spec.name,
                "version": cls.VERSION,
                "params": registry.affecting_params(cls.Params(**spec.params)),
            }
        )
    body["preprocessing"] = steps
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
    body.pop("feature_selection", None)
    if config.feature_selection is not None:
        spec = config.feature_selection
        cls = _check_adapter("selector", spec, "feature_selection")
        body["feature_selection"] = {
            "name": spec.name,
            "version": cls.VERSION,
            "params": registry.affecting_params(cls.Params(**spec.params)),
        }
    cls = _check_adapter("head", config.head, "head")
    body["head"] = {
        "name": config.head.name,
        "version": cls.VERSION,
        "params": registry.affecting_params(cls.Params(**config.head.params)),
    }
    return body


def feature_cache_key(
    config: ExperimentConfig,
    spec: AdapterSpec,
    split_hash: str,
    fitted: Sequence[Mapping[str, Any]],
) -> str:
    """Content key of one extractor's feature matrix over all windows (AD-10).

    Head, seeds and role never enter the key. `window_len` and `stride` do only for
    per-window extractors.
    """
    dataset_cls = _check_adapter("dataset", AdapterSpec(name=config.dataset), "dataset")
    steps = []
    for i, step in enumerate(config.preprocessing):
        cls = _check_adapter("preprocess", step, f"preprocessing.{i}")
        steps.append(
            {
                "name": step.name,
                "version": cls.VERSION,
                "params": registry.affecting_params(cls.Params(**step.params)),
            }
        )
    ext_cls = _check_adapter("extractor", spec, f"extractor {spec.name}")
    body: dict[str, Any] = {
        "dataset": config.dataset,
        "dataset_version": dataset_cls.VERSION,
        "split_hash": split_hash,
        "rooms": list(config.selection.rooms),
        "preprocessing": steps,
        "fitted_hash": sha256_hex(list(fitted)),
        "extractor": {
            "name": spec.name,
            "params": registry.affecting_params(ext_cls.Params(**spec.params)),
            "version": ext_cls.VERSION,
            "vendor_hash": getattr(ext_cls, "VENDOR_HASH", None),
            "weights_revision": getattr(ext_cls, "WEIGHTS_REVISION", None),
        },
    }
    if getattr(ext_cls, "granularity", None) == "per_window":
        body["window_len"] = config.window_len
        body["stride"] = config.stride
    return sha256_hex(body)


def context_cache_key(
    config: ExperimentConfig,
    spec: AdapterSpec,
    fitted: Sequence[Mapping[str, Any]],
    recording_id: str,
    source_sha256: str,
    block: tuple[int, int],
) -> str:
    """Content key of one extractor's per-block intermediates (AD-10).

    Unlike `feature_cache_key` it has no rooms, `window_len` or `stride`, so every window
    length and room set that includes this block shares the entry.
    """
    dataset_cls = _check_adapter("dataset", AdapterSpec(name=config.dataset), "dataset")
    steps = []
    for i, step in enumerate(config.preprocessing):
        cls = _check_adapter("preprocess", step, f"preprocessing.{i}")
        steps.append(
            {
                "name": step.name,
                "version": cls.VERSION,
                "params": registry.affecting_params(cls.Params(**step.params)),
            }
        )
    ext_cls = _check_adapter("extractor", spec, f"extractor {spec.name}")
    return sha256_hex(
        {
            "kind": "context",
            "dataset": config.dataset,
            "dataset_version": dataset_cls.VERSION,
            "recording_id": recording_id,
            "source_sha256": source_sha256,
            "block": list(block),
            "preprocessing": steps,
            "fitted_hash": sha256_hex(list(fitted)),
            "extractor": {
                "name": spec.name,
                "params": registry.affecting_params(ext_cls.Params(**spec.params)),
                "version": ext_cls.VERSION,
                "vendor_hash": getattr(ext_cls, "VENDOR_HASH", None),
                "weights_revision": getattr(ext_cls, "WEIGHTS_REVISION", None),
            },
        }
    )


def compute_run_id(config: ExperimentConfig, split_root: str | Path = "splits") -> str:
    """Short hash of the resolved config plus adapter `VERSION`s, excluding `seeds` (AD-5)."""
    return short_hash(resolved_body(config, split_root))


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Write the resolved config as YAML so a rerun reproduces the same `run_id`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    return path
