"""Column names, paths and schemas of everything the benchmark writes (AD-13)."""

from pathlib import Path

import pyarrow as pa

from csi_bench.core.contracts import Metrics

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


def run_dir(run_id: str, root: str | Path = "runs") -> Path:
    return Path(root) / run_id


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
