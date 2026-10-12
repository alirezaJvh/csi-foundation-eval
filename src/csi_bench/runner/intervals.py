"""Test-score confidence intervals from a run's saved predictions, for `run` and the backfill."""

import json
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from csi_bench.core import results_schema as rs
from csi_bench.core.errors import DataError
from csi_bench.core.metrics import bootstrap_intervals

_COLUMNS = ("recording_id", "split", "y_true", "y_pred")


def _test_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=list(_COLUMNS))
    table = table.filter(pc.equal(table["split"], "test"))
    return (
        np.asarray(table["recording_id"].to_pylist(), dtype=object),
        table["y_true"].to_numpy().astype(np.int64),
        table["y_pred"].to_numpy().astype(np.int64),
    )


def run_intervals(
    run_id: str,
    seeds: Iterable[int],
    class_names: Sequence[str],
    root: str | Path = "runs",
) -> dict:
    """The `intervals.test` record of a run, from the `test` rows of every seed's predictions."""
    predictions = [_test_rows(rs.predictions_path(run_id, seed, root)) for seed in sorted(seeds)]
    return bootstrap_intervals(predictions, len(class_names), class_names=class_names)


def _write_json(path: Path, payload: dict) -> None:
    """Same bytes as `run._write_json`, written through a temporary file."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def backfill_intervals(root: str | Path = "runs") -> tuple[int, int]:
    """Add `intervals.test` to every `done` run whose `metrics.json` lacks it.

    Only `metrics.json` is rewritten; runs that already have intervals are not touched.
    A run whose intervals cannot be computed is skipped with a warning on stderr.
    Returns `(filled, skipped)`.
    """
    root = Path(root)
    filled = skipped = 0
    for path in rs.iter_run_dirs(root):
        try:
            run_id = rs.parse_run_dir(path, root)[2]
            status = json.loads((path / "status.json").read_text(encoding="utf-8"))
            if not isinstance(status, dict) or status.get("status") != "done":
                continue
            metrics_file = path / "metrics.json"
            document = json.loads(metrics_file.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("metrics.json is not a mapping")
        except (OSError, ValueError, DataError) as e:
            sys.stderr.write(f"warning: {path}: cannot read run files: {e}\n")
            skipped += 1
            continue
        if "intervals" in document:
            continue
        try:
            seeds = [int(s) for s in document["seeds"]]
            class_names = [str(n) for n in document["class_names"]]
            if not seeds or not class_names:
                raise ValueError("no seeds or class names in metrics.json")
            document["intervals"] = {"test": run_intervals(run_id, seeds, class_names, root)}
            _write_json(metrics_file, document)
        except (OSError, ValueError, KeyError, TypeError, DataError) as e:
            sys.stderr.write(f"warning: {path}: no intervals: {e}\n")
            skipped += 1
            continue
        filled += 1
    return filled, skipped
