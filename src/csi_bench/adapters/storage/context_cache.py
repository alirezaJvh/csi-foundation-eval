"""npz-backed store of per-block extractor intermediates; used only by the runner (AD-10)."""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class NpzContextCache:
    """One `<key>.npz` per entry holding a single array under `values`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def get(self, key: str) -> np.ndarray | None:
        """Return the cached array, or None on absence or corruption."""
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return np.asarray(data["values"])
        except Exception as e:  # noqa: BLE001 - any unreadable entry is a miss
            logger.warning("context cache entry %s unreadable (%s); recomputing", key, e)
            return None

    def put(self, key: str, values: np.ndarray) -> None:
        """Atomically write the entry; a failure leaves neither entry nor temp file."""
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f"{key}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(handle, values=values)
            os.replace(tmp, self._path(key))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
