"""Orchestrate one experiment config end-to-end and write its run directory (AD-13)."""

import importlib.metadata
import json
import logging
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from csi_bench.adapters.storage.feature_cache import NpzFeatureCache
from csi_bench.config.experiment import (
    ExperimentConfig,
    compute_run_id,
    feature_cache_key,
    load_config,
    save_config,
)
from csi_bench.core import registry
from csi_bench.core import results_schema as rs
from csi_bench.core.contracts import FeatureMatrix, Predictions, Recording, WindowSet
from csi_bench.core.errors import ConfigError, DataError, ExtractorError
from csi_bench.core.metrics import evaluate
from csi_bench.core.ports import FeatureCache
from csi_bench.core.preprocess import run_chain
from csi_bench.core.splitplan import PARTS, find_split, load_split_plan
from csi_bench.core.windows import build_windows
from csi_bench.runner.discover import discover

PACKAGES = ("numpy", "scikit-learn", "pandas", "pyarrow", "pydantic", "PyYAML", "torch")
DEVICE = "cpu"
TORCH_THREADS = 1
DEFAULT_ROOT = "runs"
DEFAULT_CACHE_ROOT = ".cache/features"


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
            if slices is None:
                slices = [by_id[w.recording_id].csi[:, w.start : w.end, :] for w in windows]
            cls: Any = registry.get("extractor", spec.name)
            extractor = cls(cls.Params(**spec.params))
            matrix = extractor.extract(slices)
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


def _execute(
    config: ExperimentConfig,
    config_path: str | Path,
    run_id: str,
    root: str | Path,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
) -> None:
    previous = _completed_seeds(config, run_id, root)
    todo = [seed for seed in config.seeds if seed not in previous]
    if not todo:
        return
    directory = rs.run_dir(run_id, root)
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
        recordings = source.load()
        plan = load_split_plan(find_split(config.dataset, config.split), recordings)
        class_names = tuple(source.classes)

        known_rooms = {r.room for r in recordings}
        unknown = sorted(set(config.selection.rooms) - known_rooms)
        if unknown:
            raise ConfigError(
                f"selection.rooms: unknown {unknown}; dataset rooms: {sorted(known_rooms)}"
            )
        selected = [r for r in recordings if r.room in set(config.selection.rooms)]
        steps = []
        for spec in config.preprocessing:
            step_cls: Any = registry.get("preprocess", spec.name)
            steps.append(step_cls(step_cls.Params(**spec.params)))
        selected, fitted = run_chain(steps, selected, plan.blocks)
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
        matrix = _extract(
            config,
            windows,
            by_id,
            NpzFeatureCache(cache_root),
            plan.split_hash,
            fitted,
            log,
            cache_report,
        )

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
        per_seed = {
            seed: fresh[seed]
            if seed in fresh
            else {split: rs.metrics_from_dict(previous[seed][split]) for split in rs.SPLIT_NAMES}
            for seed in config.seeds
        }

        _write_json(
            rs.metrics_path(run_id, root), rs.metrics_document(per_seed, seed_names, seed_selection)
        )
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
        log.info("done in %.2fs", time.monotonic() - started)
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
) -> int:
    """Validate, execute and store one experiment; return the CLI exit code."""
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
        _execute(config, config_path, run_id, root, cache_root)
    except Exception as e:  # noqa: BLE001 - the run boundary maps every error to exit 2
        sys.stderr.write(f"error: {e}\n")
        return 2
    sys.stdout.write(f"{run_id}\n")
    return 0
