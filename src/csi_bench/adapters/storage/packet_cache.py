"""npz-backed store of one recording's per-packet features; used only by the runner (AD-10)."""

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from csi_bench.core.contracts import PacketFeatures

logger = logging.getLogger(__name__)


class NpzPacketCache:
    """One `<key>.npz` per entry holding `values`, `feature_names` and `input_stats` (JSON)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def get(self, key: str) -> PacketFeatures | None:
        """Return the cached features, or None on absence or corruption."""
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                values = np.asarray(data["values"])
                names = tuple(str(n) for n in data["feature_names"])
                stats = json.loads(str(data["input_stats"]))
            if values.ndim != 2 or values.shape[1] != len(names) or not isinstance(stats, dict):
                raise ValueError("inconsistent entry")
        except Exception as e:  # noqa: BLE001 - any unreadable entry is a miss
            logger.warning("packet cache entry %s unreadable (%s); recomputing", key, e)
            return None
        return PacketFeatures(
            values=values,
            feature_names=names,
            input_stats={str(k): float(v) for k, v in stats.items()},
        )

    def put(self, key: str, features: PacketFeatures) -> None:
        """Atomically write the entry; a failure leaves neither entry nor temp file."""
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f"{key}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(
                    handle,
                    values=features.values,
                    feature_names=np.array(features.feature_names, dtype=str),
                    input_stats=np.array(json.dumps(features.input_stats, sort_keys=True)),
                )
            os.replace(tmp, self._path(key))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
