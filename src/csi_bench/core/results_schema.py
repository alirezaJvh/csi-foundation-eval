"""Column names and dtypes of the tables the benchmark writes."""

from pathlib import Path

import pyarrow as pa

SCOPES = ("recording", "class", "room", "total")

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
