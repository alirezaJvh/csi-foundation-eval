"""npz-backed feature cache; constructed and called only by the runner (AD-10)."""

import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from csi_bench.core.contracts import FeatureMatrix

logger = logging.getLogger(__name__)


class NpzFeatureCache:
    """One `<key>.npz` per entry holding `values`, `feature_names` and `window_ids`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def get(self, key: str, window_ids: Sequence[str] | None = None) -> FeatureMatrix | None:
        """Return the cached matrix, or None on absence, corruption or stale `window_ids`."""
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                values = np.asarray(data["values"])
                names = tuple(str(n) for n in data["feature_names"])
                stored = tuple(str(w) for w in data["window_ids"])
            if values.ndim != 2 or values.shape[1] != len(names) or values.shape[0] != len(stored):
                raise ValueError("inconsistent shapes")
        except Exception as e:  # noqa: BLE001 - any unreadable entry is a miss
            logger.warning("feature cache entry %s unreadable (%s); recomputing", key, e)
            return None
        if window_ids is not None and stored != tuple(window_ids):
            logger.warning("feature cache entry %s has different windows; recomputing", key)
            return None
        return FeatureMatrix(values=values, feature_names=names)

    def put(self, key: str, matrix: FeatureMatrix, window_ids: Sequence[str]) -> None:
        """Atomically write the entry; a failure leaves neither entry nor temp file."""
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f"{key}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(
                    handle,
                    values=matrix.values,
                    feature_names=np.array(matrix.feature_names, dtype=str),
                    window_ids=np.array(list(window_ids), dtype=str),
                )
            os.replace(tmp, self._path(key))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
