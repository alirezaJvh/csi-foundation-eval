"""Expand a base config plus a `sweep:` block of dotted paths into validated configs."""

import itertools
from copy import deepcopy
from typing import Any

from csi_bench.config.experiment import ExperimentConfig, compute_run_id, config_from_dict
from csi_bench.core.errors import ConfigError


def _tokens(path: str, prefix: str = "sweep") -> list[str]:
    parts = path.split(".")
    if not path or any(not p for p in parts):
        raise ConfigError(f"{prefix}.{path}: malformed dotted path")
    return parts


def _set_path(raw: dict[str, Any], path: str, value: Any, prefix: str = "sweep") -> None:
    """Set `path` in `raw`: dict keys and list indices; missing mappings are created."""
    parts = _tokens(path, prefix)
    node: Any = raw
    for depth, part in enumerate(parts[:-1]):
        here = ".".join(parts[: depth + 1])
        if isinstance(node, list):
            node = _index(node, part, here, prefix)
        elif isinstance(node, dict):
            if part not in node or node[part] is None:
                node[part] = {}
            node = node[part]
        else:
            raise ConfigError(f"{prefix}.{path}: {here} is not a mapping or list")
    last = parts[-1]
    if isinstance(node, list):
        node[_position(node, last, path, prefix)] = value
    elif isinstance(node, dict):
        node[last] = value
    else:
        raise ConfigError(f"{prefix}.{path}: parent is not a mapping or list")


def _position(node: list[Any], part: str, path: str, prefix: str = "sweep") -> int:
    if not part.isdigit():
        raise ConfigError(
            f"{prefix}.{path}: {part!r} is not a valid index into a list of {len(node)}"
        )
    try:
        idx = int(part)
    except ValueError:  # unicode digits like "²" pass isdigit() but not int()
        raise ConfigError(
            f"{prefix}.{path}: {part!r} is not a valid index into a list of {len(node)}"
        ) from None
    if not 0 <= idx < len(node):
        raise ConfigError(
            f"{prefix}.{path}: {part!r} is not a valid index into a list of {len(node)}"
        )
    return idx


def _index(node: list[Any], part: str, path: str, prefix: str = "sweep") -> Any:
    return node[_position(node, part, path, prefix)]


def _overlap(a: str, b: str) -> bool:
    """Whether dotted paths `a` and `b` set the same field (equal, or one inside the other)."""
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def expand_sweep(raw: Any, source: str = "sweep") -> list[tuple[dict[str, Any], ExperimentConfig]]:
    """Cartesian product in file order (first key slowest), all validated, deduped by run id.

    A key starting with `@` is a paired axis: each value is a mapping of dotted paths applied
    together, so fields that must change jointly (role, split, preprocessing) are never crossed.

    `sweep` may also be a non-empty list of such mappings: each item is expanded against the
    same base, items are concatenated in file order and deduplicated by run id across items.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: config must be a mapping")
    base = {k: v for k, v in raw.items() if k != "sweep"}
    sweep = raw.get("sweep")
    if sweep is None:
        sweep = {}
    items: list[tuple[str, dict[str, Any]]]
    if isinstance(sweep, dict):
        items = [("sweep", sweep)]
    elif isinstance(sweep, list) and sweep:
        items = []
        for i, item in enumerate(sweep):
            if not isinstance(item, dict):
                raise ConfigError(f"{source}: sweep.{i}: expected a mapping of dotted paths")
            items.append((f"sweep.{i}", item))
    else:
        raise ConfigError(
            f"{source}: sweep must be a mapping of dotted paths to value lists,"
            " or a non-empty list of such mappings"
        )
    out: list[tuple[dict[str, Any], ExperimentConfig]] = []
    seen: set[str] = set()
    for prefix, item in items:
        _expand_one(base, item, prefix, source, out, seen)
    return out


def _expand_one(
    base: dict[str, Any],
    sweep: dict[str, Any],
    prefix: str,
    source: str,
    out: list[tuple[dict[str, Any], ExperimentConfig]],
    seen: set[str],
) -> None:
    """Validate one sweep mapping and append its product to `out`, skipping run ids in `seen`."""
    owners: dict[str, str] = {}  # every path an axis sets -> that axis
    for key, values in sweep.items():
        path = str(key)
        if path.startswith("@"):
            if len(path) < 2:
                raise ConfigError(f"{source}: {prefix}.{path}: paired axis needs a name")
            if not isinstance(values, list) or not values:
                raise ConfigError(
                    f"{source}: {prefix}.{path}: expected a non-empty list of mappings"
                )
            paths: set[str] = set()
            for i, entry in enumerate(values):
                if not isinstance(entry, dict) or not entry:
                    raise ConfigError(
                        f"{source}: {prefix}.{path}.{i}: expected a non-empty mapping"
                    )
                subs = [str(sub) for sub in entry]
                for j, sub_path in enumerate(subs):
                    _tokens(sub_path, prefix)
                    if sub_path == "seeds" or sub_path.startswith("seeds."):
                        raise ConfigError(f"{source}: {prefix}.{path}.{i}: seeds cannot be swept")
                    clash = next((o for o in subs[:j] if _overlap(o, sub_path)), None)
                    if clash is not None:
                        raise ConfigError(
                            f"{source}: {prefix}.{path}.{i}: {clash} and {sub_path} set the same"
                            " field"
                        )
                paths.update(subs)
        else:
            if path == "seeds" or path.startswith("seeds."):
                raise ConfigError(f"{source}: {prefix}.{path}: seeds cannot be swept")
            _tokens(path, prefix)
            if not isinstance(values, list) or not values:
                raise ConfigError(f"{source}: {prefix}.{path}: expected a non-empty list of values")
            paths = {path}
        for p in sorted(paths):
            other = next((o for o in owners if _overlap(o, p)), None)
            if other is not None:
                raise ConfigError(
                    f"{source}: {prefix}.{path}: {p} is also set by {prefix}.{owners[other]}"
                )
        owners.update(dict.fromkeys(paths, path))
    keys = [str(k) for k in sweep]
    for combo in itertools.product(*(sweep[k] for k in sweep)):
        overrides = dict(zip(keys, combo, strict=True))
        candidate = deepcopy(base)
        try:
            for path, value in overrides.items():
                if path.startswith("@"):
                    for sub_path, sub_value in value.items():
                        _set_path(candidate, str(sub_path), deepcopy(sub_value), prefix)
                else:
                    _set_path(candidate, path, deepcopy(value), prefix)
            config = config_from_dict(candidate, source)
        except ConfigError as e:
            raise ConfigError(f"{e} (overrides: {overrides})") from e
        run_id = compute_run_id(config)
        if run_id in seen:
            continue
        seen.add(run_id)
        out.append((overrides, config))
