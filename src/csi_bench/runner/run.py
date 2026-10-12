"""Orchestrate one experiment config end-to-end and write its run directory (AD-13)."""

import importlib.metadata
import json
import logging
import multiprocessing
import os
import random
import resource
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from csi_bench.adapters.storage.context_cache import NpzContextCache
from csi_bench.adapters.storage.feature_cache import NpzFeatureCache
from csi_bench.config.experiment import (
    ExperimentConfig,
    compute_run_id,
    context_cache_key,
    feature_cache_key,
    load_config,
    save_config,
)
from csi_bench.config.sweep import expand_sweep
from csi_bench.core import registry
from csi_bench.core import results_schema as rs
from csi_bench.core.contracts import FeatureMatrix, Predictions, Recording, WindowSet
from csi_bench.core.errors import ConfigError, DataError, ExtractorError
from csi_bench.core.metrics import evaluate
from csi_bench.core.ports import ContextCache, FeatureCache
from csi_bench.core.preprocess import run_chain
from csi_bench.core.splitplan import (
    PARTS,
    ROOM_HOLDOUT,
    check_split_role,
    check_split_rooms,
    find_split,
    load_split_plan,
    read_plan_params,
    read_plan_splitter,
)
from csi_bench.core.windows import build_windows
from csi_bench.runner.discover import discover
from csi_bench.runner.intervals import run_intervals

PACKAGES = ("numpy", "scikit-learn", "pandas", "pyarrow", "pydantic", "PyYAML", "torch")
DEVICE = "cpu"
TORCH_THREADS = 1
DEFAULT_ROOT = "runs"
DEFAULT_CACHE_ROOT = ".cache/features"
WORKERS_ENV = "CSI_BENCH_WORKERS"
DEFAULT_MAX_WORKERS = 4


def resolve_workers(value: int | str | None = None) -> int:
    """Runner-owned worker count: `--workers`, else `CSI_BENCH_WORKERS`, else min(4, cpus)."""
    source, raw = "--workers", value
    if raw is None:
        source, raw = f"environment variable {WORKERS_ENV}", os.environ.get(WORKERS_ENV)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return max(1, min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1))
    try:
        count = int(raw)
    except (TypeError, ValueError):
        raise ConfigError(f"{source}: expected a positive integer, got {raw!r}") from None
    if count < 1:
        raise ConfigError(f"{source}: expected a positive integer, got {raw!r}")
    return count


def _peak_rss_mb() -> tuple[float, float]:
    """This process's peak RSS and the largest peak RSS of any terminated child, in MiB.

    `RUSAGE_CHILDREN` reports the maximum over all waited-for children (cumulative
    across the run), not the sum of live workers, so the child figure is a floor.
    """
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return own / 1024, children / 1024


_BENIGN_HOUSEKEEPING = frozenset(
    {
        "ExecutorManagerThread",
        "_ExecutorManagerThread",
        "QueueFeederThread",
        "TMonitor",
        "tqdm_monitor",
    }
)


def _fork_is_safe(log: logging.Logger) -> bool:
    """Forking a multi-threaded process can deadlock; stay in-process when threads are live.

    Outlives-everything housekeeping monitors never touch extraction state, so they must
    not disable parallel extraction: loky's executor manager and queue feeder persist
    after each `joblib.Parallel` (its manager's class is
    `loky.process_executor._ExecutorManagerThread`, others identify by `name`), and
    tqdm's monitor lives for the whole process once any dependency (e.g. `mrmr`) shows
    a progress bar.
    """
    stray = [
        thread
        for thread in threading.enumerate()
        if thread is not threading.main_thread()
        and type(thread).__name__ not in _BENIGN_HOUSEKEEPING
        and thread.name not in _BENIGN_HOUSEKEEPING
    ]
    if stray:
        log.warning(
            "workers>1 requested but %d threads are live; extracting in-process to avoid "
            "a fork deadlock",
            len(stray),
        )
        return False
    return True


# Set in the parent right before the pool forks, so workers share the recordings copy-on-write.
_FORK_STATE: tuple[Any, Any, dict[str, Recording]] | None = None


def _extract_chunk(bounds: tuple[int, int]) -> FeatureMatrix:
    assert _FORK_STATE is not None
    extractor, windows, by_id = _FORK_STATE
    start, stop = bounds
    return extractor.extract(
        [by_id[w.recording_id].csi[:, w.start : w.end, :] for w in windows[start:stop]]
    )


def _chunk_bounds(n_items: int, n_chunks: int) -> list[tuple[int, int]]:
    """Contiguous, ordered, near-equal [start, stop) ranges; never more chunks than items."""
    n_chunks = max(1, min(n_chunks, n_items))
    edges = [n_items * i // n_chunks for i in range(n_chunks + 1)]
    return list(zip(edges[:-1], edges[1:], strict=True))


def _extract_blocks(
    extractor: Any,
    windows: Any,
    by_id: Any,
    block_context: Callable[[str, str], np.ndarray | None] | None = None,
    bounds: Mapping[str, Mapping[str, tuple[int, int]]] | None = None,
) -> FeatureMatrix:
    """Block-aware extraction: one call per (recording, split) block, rows in window order.

    With `block_context` and the plan's block `bounds`, the extractor gets its whole-block
    intermediates for `(recording_id, split)` and offsets are relative to the block start.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for i, w in enumerate(windows):
        groups.setdefault((w.recording_id, w.split), []).append(i)
    values = np.empty((len(windows), 0), dtype=np.float32)
    names: tuple[str, ...] | None = None
    for indices in groups.values():
        members = [windows[i] for i in indices]
        window_len = members[0].end - members[0].start
        recording_id, split = members[0].recording_id, members[0].split
        context = None
        if block_context is not None and bounds is not None:
            start, end = bounds[recording_id][split]
            context = block_context(recording_id, split)
        else:
            start = min(w.start for w in members)
            end = max(w.end for w in members)
        block = by_id[recording_id].csi[:, start:end, :]
        offsets = [w.start - start for w in members]
        if context is None:
            part = extractor.extract_blocks(block, offsets, window_len)
        else:
            part = extractor.extract_blocks(block, offsets, window_len, context=context)
        if names is None:
            names = part.feature_names
            values = np.empty((len(windows), part.values.shape[1]), dtype=np.float32)
        elif part.feature_names != names:
            raise ExtractorError("extractor returned different feature names across blocks")
        values[indices] = part.values
    if names is None:
        raise ExtractorError("no windows to extract")
    return FeatureMatrix(values=values, feature_names=names)


def _block_context_getter(
    config: ExperimentConfig,
    spec: Any,
    fitted: Sequence[dict],
    extractor: Any,
    by_id: dict[str, Recording],
    bounds: Mapping[str, Mapping[str, tuple[int, int]]],
    cache: ContextCache,
    log: logging.Logger,
) -> Callable[[str, str], np.ndarray | None]:
    """Get-or-compute one block's extractor intermediates through the runner-owned cache."""

    def get(recording_id: str, split: str) -> np.ndarray:
        rec = by_id[recording_id]
        start, end = bounds[recording_id][split]
        key = context_cache_key(
            config, spec, fitted, recording_id, rec.meta.source_sha256, (start, end)
        )
        context = cache.get(key)
        if context is not None:
            log.info("block context %s %s:%s cached", spec.name, recording_id, split)
            return context
        context = extractor.compute_context(rec.csi[:, start:end, :])
        cache.put(key, context)
        log.info("block context %s %s:%s computed", spec.name, recording_id, split)
        return context

    return get


def _extract_parallel(extractor: Any, windows: Any, by_id: Any, workers: int) -> FeatureMatrix:
    global _FORK_STATE
    bounds = _chunk_bounds(len(windows), workers)
    _FORK_STATE = (extractor, windows, by_id)
    try:
        with ProcessPoolExecutor(
            max_workers=len(bounds), mp_context=multiprocessing.get_context("fork")
        ) as pool:
            parts = list(pool.map(_extract_chunk, bounds))
    finally:
        _FORK_STATE = None
    names = parts[0].feature_names
    if any(part.feature_names != names for part in parts):
        raise ExtractorError("extractor returned different feature names across chunks")
    return FeatureMatrix(values=np.vstack([part.values for part in parts]), feature_names=names)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_status(run_id: str, root: str | Path, status: str, **extra: Any) -> None:
    payload = {"run_id": run_id, "status": status, "updated_at": _now(), **extra}
    _write_json(rs.status_path(run_id, root), payload)


def _git(args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _setup_runtime() -> None:
    """Pin device, thread count and determinism flags for the whole run (runner-owned)."""
    torch.set_default_device(DEVICE)
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global state used by third-party code
    torch.manual_seed(seed)


def runtime_info() -> dict[str, Any]:
    return {
        "device": DEVICE,
        "torch_threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "torch_version": torch.__version__,
    }


def _completed_seeds(config: ExperimentConfig, run_id: str, root: str | Path) -> dict[int, dict]:
    """Seeds of a finished run whose metrics and predictions are both on disk."""
    try:
        status = json.loads(rs.status_path(run_id, root).read_text(encoding="utf-8"))
        document = json.loads(rs.metrics_path(run_id, root).read_text(encoding="utf-8"))
        stored = document["seeds"]
    except (OSError, ValueError, KeyError):
        return {}
    if status.get("status") != "done":
        return {}
    done: dict[int, dict] = {}
    for seed in config.seeds:
        record = stored.get(str(seed))
        if (
            isinstance(record, dict)
            and set(rs.SPLIT_NAMES) <= set(record)
            and isinstance(record.get("feature_names"), list)
            and rs.predictions_path(run_id, seed, root).is_file()
        ):
            done[seed] = record
    return done


def _step_version(name: str) -> int:
    cls: Any = registry.get("preprocess", name)
    return cls.VERSION


def collect_provenance(
    config: ExperimentConfig,
    run_id: str,
    split_hash: str,
    duration: float,
    fitted: Sequence[dict],
    feature_cache: Sequence[dict] = (),
) -> dict[str, Any]:
    """Git state, package versions, split hash and duration for `provenance.json` (AD-13)."""
    status = _git(["status", "--porcelain"])
    dirty = bool(status)
    return {
        "run_id": run_id,
        "dataset": config.dataset,
        "role": config.role,
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "dirty": dirty,
            "diff": _git(["diff", "HEAD"]) if dirty else None,
        },
        "package_versions": _package_versions(),
        "split_hash": split_hash,
        "preprocessing": [
            {
                "name": spec.name,
                "version": _step_version(spec.name),
                "params": spec.params,
                "fitted": state,
            }
            for spec, state in zip(config.preprocessing, fitted, strict=True)
        ],
        "device": DEVICE,
        "runtime": runtime_info(),
        "feature_cache": list(feature_cache),
        "seeds": list(config.seeds),
        "duration_seconds": duration,
    }


def _validate_extractors(config: ExperimentConfig, recordings: Sequence[Recording]) -> None:
    antennas = {rec.csi.shape[0] for rec in recordings}
    for i, spec in enumerate(config.extractors):
        cls: Any = registry.get("extractor", spec.name)
        required = getattr(cls, "requires", {}) or {}
        wanted = required.get("n_antennas", "any")
        if isinstance(wanted, int) and any(count != wanted for count in antennas):
            raise ConfigError(
                f"extractors.{i}.name: {spec.name} requires {wanted} antennas, "
                f"got {sorted(antennas)}"
            )


def _extract(
    config: ExperimentConfig,
    windows: WindowSet,
    by_id: dict[str, Recording],
    cache: FeatureCache,
    split_hash: str,
    fitted: Sequence[dict],
    log: logging.Logger,
    cache_report: list[dict[str, Any]],
    workers: int = 1,
    context_cache: ContextCache | None = None,
    bounds: Mapping[str, Mapping[str, tuple[int, int]]] | None = None,
) -> FeatureMatrix:
    """Per extractor: load its matrix from the cache or compute it from its windows' slices."""
    window_ids = [w.window_id for w in windows]
    slices: list[np.ndarray] | None = None
    columns: list[np.ndarray] = []
    names: list[str] = []
    for spec in config.extractors:
        key = feature_cache_key(config, spec, split_hash, fitted)
        matrix = cache.get(key, window_ids)
        hit = matrix is not None
        if matrix is None:
            cls: Any = registry.get("extractor", spec.name)
            extractor = cls(cls.Params(**spec.params))
            began = time.monotonic()
            if getattr(extractor, "uses_context", False):
                block_context = None
                if (
                    context_cache is not None
                    and bounds is not None
                    and hasattr(extractor, "compute_context")
                ):
                    block_context = _block_context_getter(
                        config, spec, fitted, extractor, by_id, bounds, context_cache, log
                    )
                matrix = _extract_blocks(extractor, windows, by_id, block_context, bounds)
                path = "blocks"
            elif workers > 1 and len(windows) > 1 and _fork_is_safe(log):
                matrix = _extract_parallel(extractor, windows, by_id, workers)
                path = "parallel"
            else:
                if slices is None:
                    slices = [by_id[w.recording_id].csi[:, w.start : w.end, :] for w in windows]
                path = "slices"
                matrix = extractor.extract(slices)
            own, children = _peak_rss_mb()
            log.info(
                "extraction %s: %.2fs path=%s configured_workers=%d "
                "peak_rss_mb=%.0f max_child_peak_rss_mb=%.0f",
                spec.name,
                time.monotonic() - began,
                path,
                workers,
                own,
                children,
            )
            cache.put(key, matrix, window_ids)
        log.info("feature cache %s %s key=%s", "hit" if hit else "miss", spec.name, key)
        cache_report.append({"extractor": spec.name, "key": key, "hit": hit})
        columns.append(matrix.values)
        names.extend(matrix.feature_names)
    if len(set(names)) != len(names):
        raise ExtractorError(f"duplicate feature names across extractors: {names}")
    values = np.hstack(columns).astype(np.float32)
    if not np.isfinite(values).all():
        raise ExtractorError("features contain non-finite values")
    spread = values.astype(np.float64).std(axis=0)
    constant = np.where(spread == 0)[0]
    if values.size and constant.size:
        raise ExtractorError(f"constant feature columns: {[names[i] for i in constant]}")
    return FeatureMatrix(values=values, feature_names=tuple(names))


def _subset(matrix: FeatureMatrix, indices: list[int]) -> FeatureMatrix:
    return FeatureMatrix(values=matrix.values[indices], feature_names=matrix.feature_names)


def _predictions(
    split: str,
    windows: WindowSet,
    indices: list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Predictions:
    return Predictions(
        window_ids=tuple(windows[i].window_id for i in indices),
        y_true=y_true,
        y_pred=y_pred,
        split=split,
    )


def _write_predictions(
    path: Path,
    predictions: Sequence[Predictions],
    class_names: Sequence[str],
) -> None:
    rows = []
    for pred in predictions:
        pairs = zip(pred.window_ids, pred.y_true, pred.y_pred, strict=True)
        for window_id, y_true, y_pred in pairs:
            rows.append(
                {
                    "window_id": window_id,
                    "recording_id": window_id.rsplit("@", 1)[0],
                    "split": pred.split,
                    "y_true": int(y_true),
                    "y_pred": int(y_pred),
                    "y_true_name": class_names[int(y_true)],
                    "y_pred_name": class_names[int(y_pred)],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=rs.PREDICTIONS_TABLE_SCHEMA)
    pq.write_table(table, path)


def _build_selector(config: ExperimentConfig) -> Any | None:
    spec = config.feature_selection
    if spec is None:
        return None
    cls: Any = registry.get("selector", spec.name)
    return cls(cls.Params(**spec.params))


def _run_seed(
    config: ExperimentConfig,
    seed: int,
    matrix: FeatureMatrix,
    labels: np.ndarray,
    windows: WindowSet,
    index_by_split: dict[str, list[int]],
    class_names: Sequence[str],
    run_id: str,
    root: str | Path,
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    head_cls: Any = registry.get("head", config.head.name)
    train_i, val_i, test_i = (index_by_split[part] for part in PARTS)
    train, val, test = (_subset(matrix, rows) for rows in (train_i, val_i, test_i))
    selector = _build_selector(config)
    if selector is None:
        n_candidates = 1
    else:
        # The selector sees train and val rows only; test is only ever transformed.
        selector.fit(train, labels[train_i], val, labels[val_i])
        n_candidates = len(selector.candidates)
    if n_candidates == 0:
        raise ConfigError("feature_selection: the selector produced no candidates")
    best_head, best_index, best_score = None, 0, -1.0
    scores: list[float] = []
    for index in range(n_candidates):
        if selector is not None:
            selector.choose(index)
        head = head_cls(head_cls.Params(**config.head.params), seed)
        head.fit(
            train if selector is None else selector.transform(train),
            labels[train_i],
            val if selector is None else selector.transform(val),
            labels[val_i],
        )
        score = getattr(head, "val_score", None)
        if score is None:
            raise ConfigError(f"head {config.head.name!r} does not expose val_score")
        scores.append(float(score))
        if best_head is None or score > best_score:
            best_head, best_index, best_score = head, index, score
    assert best_head is not None
    if selector is not None:
        selector.choose(best_index)
        val, test = selector.transform(val), selector.transform(test)
    names = list(val.feature_names)
    selection = None
    if selector is not None:
        selection = {
            "chosen": best_index,
            "candidates": [
                {"n_features": len(c), "val_score": sc}
                for c, sc in zip(selector.candidates, scores, strict=True)
            ],
        }
    val_pred = _predictions("val", windows, val_i, labels[val_i], best_head.predict(val))
    test_pred = _predictions("test", windows, test_i, labels[test_i], best_head.predict(test))
    _write_predictions(rs.predictions_path(run_id, seed, root), [val_pred, test_pred], class_names)
    metrics = {"val": evaluate(val_pred, class_names), "test": evaluate(test_pred, class_names)}
    return metrics, names, selection


def _direction(config: ExperimentConfig, splitter: str) -> str | None:
    """`r1→r2` for a `room_holdout` run (read from its plan), else None."""
    if splitter != ROOM_HOLDOUT:
        return None
    try:
        params = read_plan_params(find_split(config.dataset, config.split))
        return rs.room_direction(params["train_rooms"], params["test_rooms"])
    except (DataError, KeyError, TypeError):
        return None


def _execute(
    config: ExperimentConfig,
    config_path: str | Path,
    run_id: str,
    root: str | Path,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    force: bool = False,
    workers: int = 1,
) -> None:
    previous = {} if force else _completed_seeds(config, run_id, root)
    todo = [seed for seed in config.seeds if seed not in previous]
    if not todo:
        return
    try:
        directory: Path | None = rs.run_dir(run_id, root)
    except FileNotFoundError:
        directory = None
    if directory is None or rs.parse_run_dir(directory, root)[1] == rs.UNKNOWN:
        try:
            splitter = read_plan_splitter(find_split(config.dataset, config.split))
        except DataError:
            splitter = rs.UNKNOWN  # the real error is raised and logged below, in the run dir
        slug = rs.run_slug(config.model_dump(mode="json"), _direction(config, splitter))
        target = rs.new_run_dir(root, config.dataset, splitter, slug, run_id)
        if directory is None:
            directory = target
        elif splitter != rs.UNKNOWN:  # the plan resolves now: leave the `unknown` section
            target.parent.mkdir(parents=True, exist_ok=True)
            directory = directory.rename(target)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = _now()
    log = logging.getLogger(f"csi_bench.runner.run.{run_id}")
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = logging.FileHandler(rs.log_path(run_id, root), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    log.addHandler(handler)
    try:
        _write_status(run_id, root, "pending", started_at=started_at)
        save_config(config, rs.config_path(run_id, root))
        log.info("config loaded from %s; run_id=%s", config_path, run_id)

        _write_status(run_id, root, "running", started_at=started_at)
        _setup_runtime()
        dataset_cls: Any = registry.get("dataset", config.dataset)
        source = dataset_cls(dataset_cls.Params())
        stage = time.monotonic()
        recordings = source.load()
        log.info("stage load: %.2fs", time.monotonic() - stage)
        plan = load_split_plan(find_split(config.dataset, config.split), recordings)
        check_split_role(plan.unit, config.role)
        check_split_rooms(plan.splitter, plan.params, config.selection.rooms, config.role)
        class_names = tuple(source.classes)

        known_rooms = {r.room for r in recordings}
        unknown = sorted(set(config.selection.rooms) - known_rooms)
        if unknown:
            raise ConfigError(
                f"selection.rooms: unknown {unknown}; dataset rooms: {sorted(known_rooms)}"
            )
        selected = [r for r in recordings if r.room in set(config.selection.rooms)]
        stage = time.monotonic()
        steps = []
        for spec in config.preprocessing:
            step_cls: Any = registry.get("preprocess", spec.name)
            steps.append(step_cls(step_cls.Params(**spec.params)))
        selected, fitted = run_chain(steps, selected, plan.blocks)
        log.info("stage preprocess: %.2fs", time.monotonic() - stage)
        by_id = {rec.recording_id: rec for rec in selected}
        _validate_extractors(config, selected)
        windows = build_windows(
            plan, selected, config.selection.rooms, config.window_len, config.stride
        )
        for part in PARTS:
            if not any(w.split == part for w in windows):
                raise ConfigError(
                    f"no {part} windows for rooms {config.selection.rooms} "
                    f"with window_len {config.window_len}"
                )
        log.info("windows=%d rooms=%s", len(windows), config.selection.rooms)
        cache_report: list[dict[str, Any]] = []
        stage = time.monotonic()
        matrix = _extract(
            config,
            windows,
            by_id,
            NpzFeatureCache(cache_root),
            plan.split_hash,
            fitted,
            log,
            cache_report,
            workers,
            NpzContextCache(Path(cache_root) / "context"),
            plan.blocks,
        )
        log.info("stage extract: %.2fs", time.monotonic() - stage)

        labels = np.array([by_id[w.recording_id].label for w in windows])
        index_by_split = {
            part: [i for i, w in enumerate(windows) if w.split == part] for part in PARTS
        }
        probe = _build_selector(config)
        if probe is not None:
            probe.validate(matrix.feature_names)
        fresh: dict[int, dict[str, Any]] = {}
        seed_names: dict[int, list[str]] = {
            seed: list(rec["feature_names"]) for seed, rec in previous.items()
        }
        seed_selection: dict[int, dict] = {
            seed: rec["selection"] for seed, rec in previous.items() if "selection" in rec
        }
        stage = time.monotonic()
        with joblib.parallel_config(n_jobs=workers):
            for seed in todo:
                _seed_everything(seed)
                fresh[seed], seed_names[seed], chosen = _run_seed(
                    config,
                    seed,
                    matrix,
                    labels,
                    windows,
                    index_by_split,
                    class_names,
                    run_id,
                    root,
                )
                if chosen is not None:
                    seed_selection[seed] = chosen
        log.info("stage head fit and evaluate: %.2fs", time.monotonic() - stage)
        per_seed = {
            seed: fresh[seed]
            if seed in fresh
            else {split: rs.metrics_from_dict(previous[seed][split]) for split in rs.SPLIT_NAMES}
            for seed in config.seeds
        }

        document = rs.metrics_document(per_seed, seed_names, seed_selection)
        stage = time.monotonic()
        document["intervals"] = {"test": run_intervals(run_id, config.seeds, class_names, root)}
        log.info("stage test intervals: %.2fs", time.monotonic() - stage)
        _write_json(rs.metrics_path(run_id, root), document)
        _write_json(
            rs.provenance_path(run_id, root),
            collect_provenance(
                config, run_id, plan.split_hash, time.monotonic() - started, fitted, cache_report
            ),
        )
        _write_status(
            run_id,
            root,
            "done",
            started_at=started_at,
            finished_at=_now(),
            seeds=list(config.seeds),
        )
        own, children = _peak_rss_mb()
        log.info(
            "done in %.2fs peak_rss_mb=%.0f max_child_peak_rss_mb=%.0f",
            time.monotonic() - started,
            own,
            children,
        )
    except Exception as e:
        log.error("failed: %s\n%s", e, traceback.format_exc())
        _write_status(
            run_id,
            root,
            "failed",
            started_at=started_at,
            finished_at=_now(),
            error=f"{type(e).__name__}: {e}",
            trace=traceback.format_exc(),
        )
        raise
    finally:
        log.removeHandler(handler)
        handler.close()


def run(
    config_path: str | Path,
    root: str | Path = DEFAULT_ROOT,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    workers: int | str | None = None,
) -> int:
    """Validate, execute and store one experiment; return the CLI exit code."""
    try:
        count = resolve_workers(workers)
    except ConfigError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    try:
        discover()
    except (ImportError, ConfigError) as e:
        sys.stderr.write(f"error: adapter discovery failed: {e}\n")
        return 2
    try:
        config = load_config(config_path)
        run_id = compute_run_id(config)
    except (ConfigError, DataError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    try:
        _execute(config, config_path, run_id, root, cache_root, workers=count)
    except Exception as e:  # noqa: BLE001 - the run boundary maps every error to exit 2
        sys.stderr.write(f"error: {e}\n")
        return 2
    sys.stdout.write(f"{run_id}\n")
    return 0


def _sweep_action(config: ExperimentConfig, run_id: str, root: str | Path) -> str:
    """`skip`, `run` (resume missing seeds) or `rerun` (a finished run made on a dirty tree)."""
    if len(_completed_seeds(config, run_id, root)) != len(config.seeds):
        return "run"
    try:
        provenance = json.loads(rs.provenance_path(run_id, root).read_text(encoding="utf-8"))
        was_dirty = bool(provenance["git"]["dirty"])
    except (OSError, ValueError, KeyError, TypeError):
        return "run"
    if not was_dirty:
        return "skip"
    status = _git(["status", "--porcelain"])
    return "skip" if status is None or status else "rerun"


def run_sweep(
    path: str | Path,
    force: bool = False,
    root: str | Path = DEFAULT_ROOT,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    workers: int | str | None = None,
) -> int:
    """Expand a sweep file, validate every config, then run them one at a time.

    Exit code: 0 all fine, 1 some run failed, 2 the sweep itself is invalid.
    """
    try:
        count = resolve_workers(workers)
    except ConfigError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    try:
        discover()
    except (ImportError, ConfigError) as e:
        sys.stderr.write(f"error: adapter discovery failed: {e}\n")
        return 2
    try:
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as e:
            raise ConfigError(f"{path}: {e}") from e
        expanded = expand_sweep(raw, str(path))
        planned = [(compute_run_id(config), config) for _, config in expanded]
    except (ConfigError, DataError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    failed: list[str] = []
    skipped = 0
    for run_id, config in planned:
        try:
            action = "rerun" if force else _sweep_action(config, run_id, root)
        except DataError as e:  # e.g. the run_id has several directories
            sys.stderr.write(f"error: {run_id}: {e}\n")
            sys.stdout.write(f"{run_id} failed\n")
            failed.append(run_id)
            continue
        if action == "skip":
            skipped += 1
            sys.stdout.write(f"{run_id} skipped\n")
            continue
        try:
            _execute(
                config,
                path,
                run_id,
                root,
                cache_root,
                force=action == "rerun" or force,
                workers=count,
            )
        except Exception as e:  # noqa: BLE001 - one failed run must not stop the sweep
            sys.stderr.write(f"error: {run_id}: {e}\n")
            sys.stdout.write(f"{run_id} failed\n")
            failed.append(run_id)
        else:
            sys.stdout.write(f"{run_id} done\n")
    summary = f"{len(planned) - len(failed) - skipped} of {len(planned)} runs ok"
    if skipped:
        summary += f" ({skipped} skipped)"
    sys.stdout.write(summary + "\n")
    if failed:
        sys.stdout.write(f"failed: {', '.join(failed)}\n")
        return 1
    return 0
