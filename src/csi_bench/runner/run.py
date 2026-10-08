"""Orchestrate one experiment config end-to-end and write its run directory (AD-13)."""

import importlib.metadata
import json
import logging
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

from csi_bench.config.experiment import (
    ExperimentConfig,
    compute_run_id,
    load_config,
    save_config,
)
from csi_bench.core import registry
from csi_bench.core import results_schema as rs
from csi_bench.core.contracts import FeatureMatrix, Predictions, Recording, WindowSet
from csi_bench.core.errors import ConfigError, DataError, ExtractorError
from csi_bench.core.metrics import evaluate
from csi_bench.core.splitplan import load_split_plan
from csi_bench.core.windows import build_windows
from csi_bench.runner.discover import discover

logger = logging.getLogger(__name__)

PACKAGES = ("numpy", "scikit-learn", "pandas", "pyarrow", "pydantic", "PyYAML", "torch")
DEVICE = "cpu"
DEFAULT_ROOT = "runs"
PARTS = ("train", "val", "test")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _find_split(dataset: str, split: str | None, root: str | Path = "splits") -> Path:
    """Locate the committed plan for a dataset (the same rule `describe` uses)."""
    folder = Path(root) / dataset
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if split is not None:
        for f in files:
            if f.stem == split:
                return f
        known = ", ".join(f.stem for f in files) or "none"
        raise DataError(f"no split {split} for {dataset} in {folder} (known: {known})")
    if not files:
        raise DataError(f"no split plan found in {folder}; generate one first")
    if len(files) > 1:
        hashes = ", ".join(f.stem for f in files)
        raise DataError(f"several split plans in {folder} ({hashes}); choose one with --split")
    return files[0]


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


def collect_provenance(
    config: ExperimentConfig, run_id: str, split_hash: str, duration: float
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
        "device": DEVICE,
        "seeds": list(config.seeds),
        "duration_seconds": duration,
    }


def _validate_extractors(config: ExperimentConfig, recordings: Sequence[Recording]) -> None:
    antennas = {rec.csi.shape[0] for rec in recordings}
    for i, spec in enumerate(config.extractors):
        cls: Any = registry.get("extractor", spec.name)
        if config.window_len < cls.min_window_len:
            raise ConfigError(
                f"extractors.{i}.name: window_len {config.window_len} "
                f"< min_window_len {cls.min_window_len} for {spec.name}"
            )
        required = getattr(cls, "requires", {}) or {}
        wanted = required.get("n_antennas", "any")
        if isinstance(wanted, int) and any(count != wanted for count in antennas):
            raise ConfigError(
                f"extractors.{i}.name: {spec.name} requires {wanted} antennas, "
                f"got {sorted(antennas)}"
            )


def _extract(
    config: ExperimentConfig, windows: WindowSet, by_id: dict[str, Recording]
) -> FeatureMatrix:
    """Give each extractor only its windows' packet slices, then concatenate columns."""
    slices = [by_id[w.recording_id].csi[:, w.start : w.end, :] for w in windows]
    columns: list[np.ndarray] = []
    names: list[str] = []
    for spec in config.extractors:
        cls: Any = registry.get("extractor", spec.name)
        extractor = cls(cls.Params(**spec.params))
        matrix = extractor.extract(slices)
        columns.append(matrix.values)
        names.extend(matrix.feature_names)
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
) -> dict[str, Any]:
    head_cls: Any = registry.get("head", config.head.name)
    train_i, val_i, test_i = (index_by_split[part] for part in PARTS)
    head = head_cls(head_cls.Params(**config.head.params), seed)
    head.fit(
        _subset(matrix, train_i),
        labels[train_i],
        _subset(matrix, val_i),
        labels[val_i],
    )
    val = _predictions("val", windows, val_i, labels[val_i], head.predict(_subset(matrix, val_i)))
    test_pred = head.predict(_subset(matrix, test_i))
    test = _predictions("test", windows, test_i, labels[test_i], test_pred)
    _write_predictions(rs.predictions_path(run_id, seed, root), [val, test], class_names)
    return {"val": evaluate(val, class_names), "test": evaluate(test, class_names)}


def _execute(
    config: ExperimentConfig, config_path: str | Path, run_id: str, root: str | Path
) -> None:
    directory = rs.run_dir(run_id, root)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log = logging.getLogger(f"csi_bench.runner.run.{run_id}")
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = logging.FileHandler(rs.log_path(run_id, root), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    log.addHandler(handler)
    try:
        _write_status(run_id, root, "pending", started_at=_now())
        save_config(config, rs.config_path(run_id, root))
        log.info("config loaded from %s; run_id=%s", config_path, run_id)

        _write_status(run_id, root, "running", started_at=_now())
        dataset_cls: Any = registry.get("dataset", config.dataset)
        source = dataset_cls(dataset_cls.Params())
        recordings = source.load()
        plan = load_split_plan(_find_split(config.dataset, config.split), recordings)
        class_names = tuple(source.classes)
        by_id = {rec.recording_id: rec for rec in recordings}

        selected = [r for r in recordings if r.room in set(config.selection.rooms)]
        _validate_extractors(config, selected)
        windows = build_windows(
            plan, recordings, config.selection.rooms, config.window_len, config.stride
        )
        log.info("windows=%d rooms=%s", len(windows), config.selection.rooms)
        matrix = _extract(config, windows, by_id)

        labels = np.array([by_id[w.recording_id].label for w in windows])
        index_by_split = {
            part: [i for i, w in enumerate(windows) if w.split == part] for part in PARTS
        }
        per_seed = {
            seed: _run_seed(
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
            for seed in config.seeds
        }

        _write_json(rs.metrics_path(run_id, root), rs.metrics_document(per_seed))
        _write_json(
            rs.provenance_path(run_id, root),
            collect_provenance(config, run_id, plan.split_hash, time.monotonic() - started),
        )
        _write_status(
            run_id,
            root,
            "done",
            started_at=_now(),
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
            finished_at=_now(),
            error=f"{type(e).__name__}: {e}",
            trace=traceback.format_exc(),
        )
        raise
    finally:
        log.removeHandler(handler)
        handler.close()


def run(config_path: str | Path, root: str | Path = DEFAULT_ROOT) -> int:
    """Validate, execute and store one experiment; return the CLI exit code."""
    try:
        discover()
    except (ImportError, ConfigError) as e:
        sys.stderr.write(f"error: adapter discovery failed: {e}\n")
        return 2
    try:
        config = load_config(config_path)
        run_id = compute_run_id(config)
    except ConfigError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    try:
        _execute(config, config_path, run_id, root)
    except Exception as e:  # noqa: BLE001 - the run boundary maps every error to exit 2
        sys.stderr.write(f"error: {e}\n")
        return 2
    sys.stdout.write(f"{run_id}\n")
    return 0
