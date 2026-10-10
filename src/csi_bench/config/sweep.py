"""Expand a base config plus a `sweep:` block of dotted paths into validated configs."""

import itertools
from copy import deepcopy
from typing import Any

from csi_bench.config.experiment import ExperimentConfig, compute_run_id, config_from_dict
from csi_bench.core.errors import ConfigError


def _tokens(path: str) -> list[str]:
    parts = path.split(".")
    if not path or any(not p for p in parts):
        raise ConfigError(f"sweep.{path}: malformed dotted path")
    return parts


def _set_path(raw: dict[str, Any], path: str, value: Any) -> None:
    """Set `path` in `raw`: dict keys and list indices; missing mappings are created."""
    parts = _tokens(path)
    node: Any = raw
    for depth, part in enumerate(parts[:-1]):
        here = ".".join(parts[: depth + 1])
        if isinstance(node, list):
            node = _index(node, part, here)
        elif isinstance(node, dict):
            if part not in node or node[part] is None:
                node[part] = {}
            node = node[part]
        else:
            raise ConfigError(f"sweep.{path}: {here} is not a mapping or list")
    last = parts[-1]
    if isinstance(node, list):
        node[_position(node, last, path)] = value
    elif isinstance(node, dict):
        node[last] = value
    else:
        raise ConfigError(f"sweep.{path}: parent is not a mapping or list")


def _position(node: list[Any], part: str, path: str) -> int:
    if not part.isdigit():
        raise ConfigError(f"sweep.{path}: {part!r} is not a valid index into a list of {len(node)}")
    try:
        idx = int(part)
    except ValueError:  # unicode digits like "²" pass isdigit() but not int()
        raise ConfigError(
            f"sweep.{path}: {part!r} is not a valid index into a list of {len(node)}"
        ) from None
    if not 0 <= idx < len(node):
        raise ConfigError(f"sweep.{path}: {part!r} is not a valid index into a list of {len(node)}")
    return idx


def _index(node: list[Any], part: str, path: str) -> Any:
    return node[_position(node, part, path)]


def expand_sweep(raw: Any, source: str = "sweep") -> list[tuple[dict[str, Any], ExperimentConfig]]:
    """Cartesian product in file order (first key slowest), all validated, deduped by run id."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: config must be a mapping")
    base = {k: v for k, v in raw.items() if k != "sweep"}
    sweep = raw.get("sweep")
    if sweep is None:
        sweep = {}
    if not isinstance(sweep, dict):
        raise ConfigError(f"{source}: sweep must be a mapping of dotted paths to value lists")
    for key, values in sweep.items():
        path = str(key)
        if path == "seeds" or path.startswith("seeds."):
            raise ConfigError(f"{source}: sweep.{path}: seeds cannot be swept")
        _tokens(path)
        if not isinstance(values, list) or not values:
            raise ConfigError(f"{source}: sweep.{path}: expected a non-empty list of values")
    keys = [str(k) for k in sweep]
    out: list[tuple[dict[str, Any], ExperimentConfig]] = []
    seen: set[str] = set()
    for combo in itertools.product(*(sweep[k] for k in sweep)):
        overrides = dict(zip(keys, combo, strict=True))
        candidate = deepcopy(base)
        try:
            for path, value in overrides.items():
                _set_path(candidate, path, deepcopy(value))
            config = config_from_dict(candidate, source)
        except ConfigError as e:
            raise ConfigError(f"{e} (overrides: {overrides})") from e
        run_id = compute_run_id(config)
        if run_id in seen:
            continue
        seen.add(run_id)
        out.append((overrides, config))
    return out
