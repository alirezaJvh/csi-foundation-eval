"""`csi-bench migrate-runs`: move legacy `runs/<run_id>/` directories into the layout.

Then fill `intervals.test` into the `metrics.json` of every finished run that lacks it,
from its saved predictions (no retraining); nothing else in a run is rewritten.

Idempotent: directories are only renamed, never deleted, and a second pass finds nothing
to move or fill. Copies of a `run_id` that lose the tie are parked
under `<root>/_duplicates/`, which the report ignores.
"""

import json
import os
import re
import sys
from pathlib import Path

import yaml

from csi_bench.core import results_schema as rs
from csi_bench.core.errors import DataError
from csi_bench.core.splitplan import find_split, read_plan_splitter
from csi_bench.runner.intervals import backfill_intervals

_LEGACY_NAME = re.compile(r"^[0-9a-f]{12}$")


def find_legacy_dirs(root: str | Path = "runs") -> list[Path]:
    """Legacy run dirs anywhere under `root`, outside `_`-prefixed folders and the layout."""
    root = Path(root)
    found: list[Path] = []
    if not root.is_dir():
        return found
    for current, dirs, _ in os.walk(root):
        here = Path(current)
        keep = []
        for name in sorted(dirs):
            path = here / name
            if name.startswith("_") or path.is_symlink():
                continue
            try:
                rs.parse_run_dir(path, root)
                continue  # a layout run dir: never look inside
            except DataError:
                pass
            if _LEGACY_NAME.match(name) and (path / "config.yaml").is_file():
                found.append(path)
                continue
            keep.append(name)
        dirs[:] = keep
    return sorted(found)


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a mapping")
    return data


def _finished_at(path: Path) -> str:
    try:
        return str(_read_json(path / "status.json").get("finished_at") or "")
    except (OSError, ValueError):
        return ""


def _placement(path: Path, split_root: str | Path) -> tuple[str, str, str]:
    """`(dataset, splitter, slug)` of a legacy run; unresolvable parts become `unknown`."""
    try:
        config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
        dataset = str(config["dataset"])
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return rs.UNKNOWN, rs.UNKNOWN, rs.UNKNOWN
    try:
        slug = rs.run_slug(config)
    except (KeyError, TypeError, AttributeError):
        slug = rs.UNKNOWN
    try:
        split_hash = _read_json(path / "provenance.json").get("split_hash")
    except (OSError, ValueError):
        split_hash = None
    candidates = [s for s in (split_hash, config.get("split")) if s] + [None]
    for split in candidates:
        try:
            return dataset, read_plan_splitter(find_split(dataset, split, split_root)), slug
        except DataError:
            continue
    return dataset, rs.UNKNOWN, slug


def _park(path: Path, root: Path) -> Path:
    folder = rs.duplicates_dir(root, path.parent.relative_to(root))
    target = folder / path.name
    n = 1
    while target.exists():
        target = folder / f"{path.name}.{n}"
        n += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)
    return target


def migrate_runs(root: str | Path = "runs", split_root: str | Path = "splits") -> int:
    """Move legacy run dirs into the layout, park duplicates, backfill intervals, rebuild index."""
    from csi_bench.runner.report import refresh_index

    root = Path(root)
    legacy = find_legacy_dirs(root)
    in_layout = len(rs.iter_run_dirs(root))
    by_id: dict[str, list[Path]] = {}
    for path in legacy:
        by_id.setdefault(path.name, []).append(path)
    moved: list[tuple[Path, Path]] = []
    parked: list[tuple[Path, Path]] = []
    try:
        for run_id, copies in sorted(by_id.items()):
            try:
                rs.run_dir(run_id, root)
                keeper = None  # a layout dir already exists: every legacy copy is parked
            except FileNotFoundError:
                keeper = max(sorted(copies), key=_finished_at)  # newest; ties: first path
            for path in copies:
                if path is keeper:
                    dataset, splitter, slug = _placement(path, split_root)
                    target = rs.new_run_dir(root, dataset, splitter, slug, run_id)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(target)
                    moved.append((path, target))
                else:
                    parked.append((path, _park(path, root)))
        code = 0
    except (OSError, DataError) as e:
        sys.stderr.write(f"error: {e}\n")
        code = 2
    for source, target in parked:
        sys.stdout.write(f"parked duplicate {source} -> {target}\n")
    sys.stdout.write(
        f"moved {len(moved)}, parked {len(parked)}, skipped {in_layout} already in the layout\n"
    )
    filled, failed = backfill_intervals(root)
    sys.stdout.write(
        f"filled confidence intervals for {filled} runs"
        + (f", {failed} could not be filled (see warnings)" if failed else "")
        + "\n"
    )
    if moved or parked or filled:
        refresh_index(root)
    return code
