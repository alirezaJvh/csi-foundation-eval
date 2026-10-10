"""Read-only access to finished runs for notebooks (AD-13). Nothing here writes."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from csi_bench.core import results_schema as rs

DEFAULT_ROOT = "runs"


def load_index(root: str | Path = DEFAULT_ROOT) -> pd.DataFrame:
    """The run index written by `csi-bench report`, one row per run."""
    path = rs.index_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; run `csi-bench report` first")
    return pd.read_parquet(path)


def load_metrics(run_id: str, root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    """The run's `metrics.json` document."""
    return json.loads(rs.metrics_path(run_id, root).read_text(encoding="utf-8"))


def load_predictions(run_id: str, seed: int, root: str | Path = DEFAULT_ROOT) -> pd.DataFrame:
    """The run's per-window predictions for one seed."""
    return pd.read_parquet(rs.predictions_path(run_id, seed, root))
