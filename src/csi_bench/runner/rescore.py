"""Rerun stored runs to record their tuning-stage val score, and compare before with after.

`rescore-runs` copies every old `metrics.json`, `provenance.json` and predictions file outside
the store first, then reruns each run from its own `config.yaml`. `rescore-report` compares that
backup with the rescored store. Both refuse a dirty tree or dirty runs unless `--allow-dirty`
is given, because `report` hides dirty runs.
"""

import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from shutil import copy2
from typing import Any

from csi_bench.config.experiment import compute_run_id, load_config
from csi_bench.core import results_schema as rs
from csi_bench.core.errors import ConfigError
from csi_bench.runner.discover import discover
from csi_bench.runner.report import (
    _best,
    _cell,
    _config_row,
    _grid,
    _index_row,
    _read_json,
    _read_scores,
    _section_key,
)
from csi_bench.runner.run import DEFAULT_CACHE_ROOT, _execute, _git, resolve_workers

DEFAULT_BACKUP = "results/metrics-before-4-5"
_BACKED_UP = (rs.METRICS_FILE, rs.PROVENANCE_FILE, "seeds/*/predictions.parquet")


def _check_backup(root: Path, backup: Path) -> None:
    if backup.resolve().is_relative_to(root.resolve()):
        raise ConfigError(f"--backup {backup} must be outside --runs {root}")


def _copy_once(source: Path, target: Path) -> None:
    """Copy unless `target` exists; a copy cut short never takes the name of a real backup."""
    if target.exists() or not source.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial")
    copy2(source, partial)
    partial.replace(target)


def _scored(path: Path) -> bool:
    try:
        document = _read_json(path / rs.METRICS_FILE)
        return "macro_f1" in document["mean_std"][rs.TUNING_VAL]
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _back_up(paths: list[Path], root: Path, backup: Path) -> None:
    """Copy each run's old files; a run that has the score has no old files left to protect.

    Copying it would store new metrics as the old ones, and the comparison would then pass.
    """
    for path in paths:
        if _scored(path):
            continue
        for pattern in _BACKED_UP:
            for source in sorted(path.glob(pattern)):
                _copy_once(source, backup / source.relative_to(root))


def _has_score(path: Path) -> bool:
    """A finished run with the score; an interrupted rerun is not `done`, so it runs again."""
    try:
        return _read_json(path / "status.json")["status"] == "done" and _scored(path)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _was_dirty(path: Path) -> bool:
    try:
        return bool(_read_json(path / rs.PROVENANCE_FILE)["git"]["dirty"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _tree_is_dirty() -> bool:
    return bool(_git(["status", "--porcelain"]))


def rescore_runs(
    root: str | Path = "runs",
    backup: str | Path = DEFAULT_BACKUP,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    workers: int | str | None = None,
    allow_dirty: bool = False,
) -> int:
    """Back up, then rerun one at a time every run lacking the score.

    A scored run made on a dirty tree is rerun too when the tree is clean, so an edit during
    the rescore costs only the runs it touched.

    Exit code: 0 all fine, 1 some run failed or was refused, 2 nothing was rerun because of a
    bad argument, no runs under `root`, a dirty tree or a failed backup.
    """
    root, backup = Path(root), Path(backup)
    try:
        count = resolve_workers(workers)
        _check_backup(root, backup)
        discover()
    except (ImportError, ConfigError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    dirty_message = (
        "error: the git tree is dirty: rescored runs would be marked dirty and `report` hides "
        "them; commit first, or pass --allow-dirty for a sample outside the real store\n"
    )
    tree_dirty = _tree_is_dirty()
    if tree_dirty and not allow_dirty:
        sys.stderr.write(dirty_message)
        return 2
    paths = rs.iter_run_dirs(root)
    if not paths:
        sys.stderr.write(f"error: no runs under {root}\n")
        return 2
    try:
        _back_up(paths, root, backup)
    except OSError as e:
        sys.stderr.write(f"error: cannot back up to {backup}: {e}\n")
        return 2
    if not allow_dirty and _tree_is_dirty():  # e.g. a --backup that git does not ignore
        sys.stderr.write(dirty_message)
        return 2
    failed: list[str] = []
    skipped = unfinished = rerun = 0
    for position, path in enumerate(paths, start=1):
        run_id = rs.parse_run_dir(path, root)[2]
        if not (path / rs.METRICS_FILE).is_file():
            unfinished += 1  # it never finished, so there is no score to fill in
            continue
        try:
            if _has_score(path) and (tree_dirty or not _was_dirty(path)):
                skipped += 1
                continue
            config_file = rs.config_path(run_id, root)
            config = load_config(config_file)
            recomputed = compute_run_id(config)
            if recomputed != run_id:
                raise ConfigError(f"recomputed run_id {recomputed} differs from its directory")
            began = time.monotonic()
            _execute(config, config_file, run_id, root, cache_root, force=True, workers=count)
        except Exception as e:  # noqa: BLE001 - one failed run must not stop the rescore
            sys.stderr.write(f"error: {run_id}: {e}\n")
            sys.stdout.write(f"[{position}/{len(paths)}] {run_id} failed\n")
            failed.append(run_id)
        else:
            rerun += 1
            seconds = time.monotonic() - began
            sys.stdout.write(f"[{position}/{len(paths)}] {run_id} done {seconds:.1f}s\n")
        sys.stdout.flush()
    summary = f"{rerun} rerun, {skipped} already scored, {len(failed)} failed"
    if unfinished:
        summary += f", {unfinished} never finished (left alone)"
    sys.stdout.write(f"{summary}; backup in {backup}\n")
    if failed:
        sys.stdout.write(f"failed: {', '.join(failed)}\n")
        return 1
    return 0


def _old_best(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Replay the retired in-sample rule on backup rows, to show what `report` used to pick."""
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["method"], r["room_setting"])
        if key not in best or r["val_macro_f1_mean"] > best[key]["val_macro_f1_mean"]:
            best[key] = r
    return best


def _cells(
    rows: list[dict],
    pick: Callable[[list[dict]], dict],
    split_root: str | Path,
    gaps: dict[tuple[str, str], int | None],
) -> dict[tuple[str, ...], dict]:
    """The row `pick` selects per (dataset, section, role, method, room), as `report` groups:
    the section is the splitter, or `time_block, gap_packets <n>` for a non-default gap.
    """
    tables: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        tables[(*_section_key(r, split_root, gaps), r["role"])].append(r)
    return {
        (*table, *cell): row
        for table, members in tables.items()
        for cell, row in pick(members).items()
    }


def _test_differences(old: Any, new: Any) -> list[str]:
    """What differs in the test metrics of one run: per seed, the means, the intervals."""
    try:
        seeds = sorted(set(old["seeds"]) | set(new["seeds"]))
        found = [
            f"seed {seed}"
            for seed in seeds
            if old["seeds"].get(seed, {}).get("test") != new["seeds"].get(seed, {}).get("test")
        ]
        if old["mean_std"]["test"] != new["mean_std"]["test"]:
            found.append("mean_std")
        old_intervals = old.get("intervals", {}).get("test")
        new_intervals = new.get("intervals", {}).get("test")
        if old_intervals and old_intervals != new_intervals:
            found.append("intervals")
    except (AttributeError, KeyError, TypeError) as e:
        return [f"unreadable ({e!r})"]
    return found


def _changed_cells(old: dict, new: dict) -> dict[tuple[str, str], list[list[str]]]:
    changed: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    for key in sorted(set(old) | set(new)):
        before, after = old.get(key), new.get(key)
        same_run = before and after and before["run_id"] == after["run_id"]
        if same_run and _cell(before) == _cell(after):
            continue
        dataset, splitter, role, method, room = key
        changed[dataset, splitter].append(
            [
                role,
                method,
                room,
                before["run_id"] if before else "-",
                after["run_id"] if after else "-",
                _cell(before) if before else "-",
                _cell(after) if after else "-",
                before["variant"] if before else "-",
                after["variant"] if after else "-",
            ]
        )
    return changed


def rescore_report(
    root: str | Path = "runs",
    backup: str | Path = DEFAULT_BACKUP,
    out: str | Path | None = None,
    split_root: str | Path = "splits",
    allow_dirty: bool = False,
) -> int:
    """Compare backup and store. Exit code: 0 clean, 1 a check failed, 2 nothing to compare.

    Dirty runs are hidden from the after column and fail the check, as `report` hides them;
    `allow_dirty` shows them and passes, for a sample made on a dirty tree.
    """
    root, backup = Path(root), Path(backup)
    try:
        _check_backup(root, backup)
        if not backup.is_dir():
            raise ConfigError(f"--backup {backup}: no such directory")
    except ConfigError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    store = {rs.parse_run_dir(p, root)[2]: p for p in rs.iter_run_dirs(root)}
    backups = [(rs.parse_run_dir(p, backup)[2], p) for p in rs.iter_run_dirs(backup)]
    if not store and not backups:
        sys.stderr.write(f"error: nothing to compare in {root} and {backup}\n")
        return 2
    before_ids = {run_id for run_id, _ in backups}

    old_rows: list[dict] = []
    mismatches: list[str] = []
    for run_id, path in backups:
        if run_id not in store:
            continue
        row = _config_row(store[run_id], root, split_root)
        if row is not None:
            _read_scores(row, path / rs.METRICS_FILE, path / rs.PROVENANCE_FILE)
            old_rows.append(row)
        try:
            old_doc = _read_json(path / rs.METRICS_FILE)
            new_doc = _read_json(store[run_id] / rs.METRICS_FILE)
            differences = _test_differences(old_doc, new_doc)
        except (OSError, ValueError) as e:
            differences = [f"unreadable ({e})"]
        if differences:
            mismatches.append(f"{run_id}: {', '.join(differences)}")
    # `iter_run_dirs` order, as `report` reads it: on a tie the first run stays selected
    new_rows = [row for row in (_index_row(p, root, split_root) for p in store.values()) if row]

    only_before = sorted(before_ids - set(store))
    only_after = sorted(set(store) - before_ids)
    # Also catches a run `_index_row` could not read: it has no row, so it is never scored.
    unscored = sorted(set(store) - {r["run_id"] for r in new_rows if _is_scored_done(r)})
    dirty = sum(1 for r in new_rows if r["dirty"])
    gaps: dict[tuple[str, str], int | None] = {}
    old_best = _cells(
        [
            r
            for r in old_rows
            if r["split_hash"] and r["val_macro_f1_mean"] is not None and not r["dirty"]
        ],
        _old_best,
        split_root,
        gaps,
    )
    new_best = _cells(
        [
            r
            for r in new_rows
            if _is_scored_done(r) and r["split_hash"] and (allow_dirty or not r["dirty"])
        ],
        _best,
        split_root,
        gaps,
    )
    changed = _changed_cells(old_best, new_best)

    lines = ["# Selection before and after the tuning-stage val score", "", "## Checks", ""]
    lines += [
        f"- runs before: {len(backups)}",
        f"- runs after: {len(store)}",
        f"- ids only before: {len(only_before)}{_listed(only_before)}",
        f"- ids only after: {len(only_after)}{_listed(only_after)}",
        f"- runs not done with the score: {len(unscored)}{_listed(unscored)}",
        f"- dirty after the rescore (`report` hides them): {dirty}",
        f"- test-metric mismatches: {len(mismatches)}",
        f"- cells changed: {sum(len(v) for v in changed.values())} of "
        f"{len(set(old_best) | set(new_best))}",
    ]
    if mismatches:
        lines += ["", "## Test-metric mismatches", ""] + [f"- {m}" for m in mismatches]
    sections = {key[:2] for key in (*old_best, *new_best)}
    header = [
        "role",
        "method",
        "room",
        "run before",
        "run after",
        "score before",
        "score after",
        "variant before",
        "variant after",
    ]
    for dataset, splitter in sorted(sections):
        lines += ["", f"## {dataset} / {splitter}", ""]
        body = changed.get((dataset, splitter))
        lines += ["no cell changed"] if not body else ["```", *_grid(header, body), "```"]
    text = "\n".join(lines) + "\n"
    if out is None:
        sys.stdout.write(text)
    else:
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(text, encoding="utf-8")
        except OSError as e:
            sys.stderr.write(f"error: cannot write {out}: {e}\n")
            return 2
    blocking_dirty = 0 if allow_dirty else dirty
    failed = bool(mismatches or only_before or only_after or unscored or blocking_dirty)
    if failed:
        sys.stderr.write(
            f"note: checks failed: {len(mismatches)} test-metric mismatches, "
            f"{len(only_before) + len(only_after)} ids differ, "
            f"{len(unscored)} not done with the score, {blocking_dirty} dirty\n"
        )
    return 1 if failed else 0


def _is_scored_done(row: dict) -> bool:
    return row["status"] == "done" and row[f"{rs.TUNING_SCORE}_mean"] is not None


def _listed(ids: list[str], limit: int = 20) -> str:
    if not ids:
        return ""
    more = f" and {len(ids) - limit} more" if len(ids) > limit else ""
    return f" ({', '.join(ids[:limit])}{more})"
