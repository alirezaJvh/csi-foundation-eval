"""Column names, paths and schemas of everything the benchmark writes (AD-13)."""

import glob
import re
from pathlib import Path

import pyarrow as pa

from csi_bench.core.contracts import Metrics
from csi_bench.core.errors import DataError

SCOPES = ("recording", "class", "room", "total")

SPLIT_NAMES = ("val", "test")
STATUS_STATES = ("pending", "running", "done", "failed")

DATASET_TABLE_SCHEMA = pa.schema(
    [
        ("scope", pa.string()),
        ("recording_id", pa.string()),
        ("room", pa.string()),
        ("class", pa.string()),
        ("n_packets", pa.int64()),
        ("downsample_factor", pa.int64()),
        ("train", pa.int64()),
        ("val", pa.int64()),
        ("test", pa.int64()),
        ("split_hash", pa.string()),
    ]
)


def dataset_table_path(dataset: str, root: str | Path = "runs") -> Path:
    return Path(root) / "_dataset" / f"{dataset}.parquet"


PREDICTIONS_TABLE_SCHEMA = pa.schema(
    [
        ("window_id", pa.string()),
        ("recording_id", pa.string()),
        ("split", pa.string()),
        ("y_true", pa.int64()),
        ("y_pred", pa.int64()),
        ("y_true_name", pa.string()),
        ("y_pred_name", pa.string()),
    ]
)


RUN_ID_SEPARATOR = "--"
UNKNOWN = "unknown"
DUPLICATES = "_duplicates"
_SLUG_UNSAFE = re.compile(r"[^a-z0-9_+-]")


DIRECTION = "→"


def run_slug(config: dict, rooms: str | None = None) -> str:
    """`<extractors>-<selector>-<head>-<rooms>-w<window_len>`, sanitised to `[a-z0-9_+-]`.

    `rooms` overrides the room part (a `room_holdout` direction like `r1→r2`, written
    `r1_to_r2`). Only for reading the store by eye: `run_id` stays the sole identity.
    """
    extractors = "+".join(sorted(e["name"] for e in config["extractors"]))
    selector = (config.get("feature_selection") or {}).get("name", "none")
    if rooms is None:
        rooms = room_setting(config["selection"]["rooms"])
    rooms = rooms.replace(DIRECTION, "_to_")
    raw = f"{extractors}-{selector}-{config['head']['name']}-{rooms}-w{config['window_len']}"
    return _SLUG_UNSAFE.sub("_", raw.lower())


def new_run_dir(root: str | Path, dataset: str, splitter: str, slug: str, run_id: str) -> Path:
    """Where a run with no directory yet is created: `<root>/<dataset>/<splitter>/<slug>--<id>`."""
    return Path(root) / dataset / splitter / f"{slug}{RUN_ID_SEPARATOR}{run_id}"


def duplicates_dir(root: str | Path, origin: str | Path) -> Path:
    """Where losing copies of a run from `<root>/<origin>` are parked; the report ignores it."""
    return Path(root) / DUPLICATES / origin


def parse_run_dir(path: str | Path, root: str | Path = "runs") -> tuple[str, str, str]:
    """`(dataset, splitter, run_id)` of a layout run directory; `DataError` for anything else."""
    try:
        parts = Path(path).relative_to(Path(root)).parts
    except ValueError as e:
        raise DataError(f"{path}: not under {root}") from e
    if len(parts) != 3 or parts[0].startswith("_") or RUN_ID_SEPARATOR not in parts[2]:
        raise DataError(f"{path}: not a run directory <dataset>/<splitter>/<slug>--<run_id>")
    run_id = parts[2].rpartition(RUN_ID_SEPARATOR)[2]
    if not run_id:
        raise DataError(f"{path}: empty run_id")
    return parts[0], parts[1], run_id


def iter_run_dirs(root: str | Path = "runs") -> list[Path]:
    """Every layout run directory under `root`, sorted; `_`-prefixed folders are skipped."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for path in root.glob(f"*/*/*{RUN_ID_SEPARATOR}*"):
        if not path.is_dir():
            continue
        try:
            parse_run_dir(path, root)
        except DataError:
            continue
        found.append(path)
    return sorted(found)


def run_dir(run_id: str, root: str | Path = "runs") -> Path:
    """The existing directory of `run_id`; `FileNotFoundError` if none, `DataError` if several."""
    root = Path(root)
    matches = []
    if root.is_dir():
        pattern = f"*/*/*{RUN_ID_SEPARATOR}{glob.escape(run_id)}"
        for path in root.glob(pattern):
            try:
                found = parse_run_dir(path, root)[2]
            except DataError:
                continue
            if found == run_id and path.is_dir():
                matches.append(path)
    if not matches:
        raise FileNotFoundError(f"no run directory for {run_id} under {root}")
    if len(matches) > 1:
        listed = ", ".join(str(p) for p in sorted(matches))
        raise DataError(f"run {run_id} has several directories: {listed}")
    return matches[0]


def config_path(run_id: str, root: str | Path = "runs") -> Path:
    return run_dir(run_id, root) / "config.yaml"


def status_path(run_id: str, root: str | Path = "runs") -> Path:
    return run_dir(run_id, root) / "status.json"


def metrics_path(run_id: str, root: str | Path = "runs") -> Path:
    return run_dir(run_id, root) / "metrics.json"


def provenance_path(run_id: str, root: str | Path = "runs") -> Path:
    return run_dir(run_id, root) / "provenance.json"


def log_path(run_id: str, root: str | Path = "runs") -> Path:
    return run_dir(run_id, root) / "run.log"


def predictions_path(run_id: str, seed: int, root: str | Path = "runs") -> Path:
    return run_dir(run_id, root) / "seeds" / str(seed) / "predictions.parquet"


def metrics_as_dict(metrics: Metrics) -> dict:
    return {
        "split": metrics.split,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "per_class": metrics.per_class,
        "confusion_matrix": [list(row) for row in metrics.confusion_matrix],
        "class_names": list(metrics.class_names),
    }


def metrics_from_dict(record: dict) -> Metrics:
    """Inverse of `metrics_as_dict`."""
    return Metrics(
        split=record["split"],
        accuracy=record["accuracy"],
        macro_f1=record["macro_f1"],
        per_class=record["per_class"],
        confusion_matrix=tuple(tuple(row) for row in record["confusion_matrix"]),
        class_names=tuple(record["class_names"]),
    )


def _mean_std(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return [mean, variance**0.5]


def metrics_document(
    per_seed: dict[int, dict[str, Metrics]],
    feature_names: dict[int, list[str]] | None = None,
    selection: dict[int, dict] | None = None,
) -> dict:
    """Per-seed metrics plus mean ± std per split, in the `metrics.json` schema.

    `feature_names` maps a seed to the columns its head saw; each is stored as
    `seeds.<seed>.feature_names`. `selection` maps a seed to the selector's candidates
    (`n_features`, `val_score`) and the chosen index; each is stored as `seeds.<seed>.selection`.
    """
    seeds: dict[str, dict] = {
        str(seed): {split: metrics_as_dict(m) for split, m in splits.items()}
        for seed, splits in per_seed.items()
    }
    for seed, names in (feature_names or {}).items():
        seeds[str(seed)]["feature_names"] = list(names)
    for seed, chosen in (selection or {}).items():
        seeds[str(seed)]["selection"] = chosen
    class_names = next(iter(next(iter(per_seed.values())).values())).class_names if per_seed else ()
    aggregate: dict[str, dict] = {}
    for split in SPLIT_NAMES:
        present = [splits[split] for splits in per_seed.values() if split in splits]
        aggregate[split] = {
            "accuracy": _mean_std([m.accuracy for m in present]),
            "macro_f1": _mean_std([m.macro_f1 for m in present]),
        }
    return {
        "class_names": list(class_names),
        "seeds": seeds,
        "mean_std": aggregate,
    }


ROOM_SETTINGS = ("r1", "r2", "mixed")

# `intervals.test` of `metrics.json` in the index; null for a run without intervals.
INTERVAL_FIELDS = (
    ("test_macro_f1_lo", pa.float64()),
    ("test_macro_f1_hi", pa.float64()),
    ("test_accuracy_lo", pa.float64()),
    ("test_accuracy_hi", pa.float64()),
    ("single_recording_classes", pa.string()),
)

INDEX_TABLE_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("status", pa.string()),
        ("role", pa.string()),
        ("dataset", pa.string()),
        ("splitter", pa.string()),
        ("split_hash", pa.string()),
        ("dirty", pa.bool_()),
        ("method", pa.string()),
        ("room_setting", pa.string()),
        ("window_len", pa.int64()),
        ("stride", pa.int64()),
        ("variant", pa.string()),
        ("n_seeds", pa.int64()),
        ("val_macro_f1_mean", pa.float64()),
        ("val_macro_f1_std", pa.float64()),
        ("val_accuracy_mean", pa.float64()),
        ("val_accuracy_std", pa.float64()),
        ("test_macro_f1_mean", pa.float64()),
        ("test_macro_f1_std", pa.float64()),
        ("test_accuracy_mean", pa.float64()),
        ("test_accuracy_std", pa.float64()),
        *INTERVAL_FIELDS,
    ]
)


def index_path(root: str | Path = "runs") -> Path:
    return Path(root) / "index.parquet"


def room_setting(rooms: list[str]) -> str:
    """`r1`, `r2` or `mixed` for the two-room set; any other set is its sorted names joined."""
    chosen = sorted(set(rooms))
    if chosen == ["r1", "r2"]:
        return "mixed"
    return "+".join(chosen)


def room_direction(train_rooms: list[str], test_rooms: list[str]) -> str:
    """The report column of a `room_holdout` run, e.g. `r1→r2`."""
    return f"{room_setting(train_rooms)}{DIRECTION}{room_setting(test_rooms)}"
