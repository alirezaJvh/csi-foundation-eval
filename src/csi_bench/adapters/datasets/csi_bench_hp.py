"""CSI-Bench (NeurIPS 2025 D&B): HAR recordings of the three HP devices, complex CSI.

Layout under the root (`Multitask/sub_Human_mat` of the release zip):
`user_<U>/act_<activity>/env_<E>/device_HP/<name>.mat`, each holding a `csi_trace`
struct whose `csi` is complex `(2 tx, 2 rx, 58 subcarriers, packets)`. Only transmitter 0
is kept, giving two receive chains `(2, packets, 58)`. Other activities and devices are
skipped.
"""

import hashlib
import io
import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.io import loadmat

from csi_bench.core.contracts import Recording, RecordingMeta
from csi_bench.core.errors import DataError
from csi_bench.core.npy_map import is_exhaustion, map_npy
from csi_bench.core.registry import register

logger = logging.getLogger(__name__)

CLASSES = ("jumping", "running", "seated_breathing", "walking", "waving_hand")
_ACTIVITIES = {
    "jumping": 0,
    "running": 1,
    "seated-breathing": 2,
    "walking-case1": 3,
    "walking-case2": 3,
    "walking-case3": 3,
    "walking-case4": 3,
    "wavinghand": 4,
}
_N_TX = 2
_N_RX = 2
_N_SUBCARRIERS = 58
_PATTERN = "user_*/act_*/env_*/device_HP/*.mat"
_PREFIX = "csi_bench_hp"


def _convert(raw: bytes, name: str) -> np.ndarray:
    """`csi_trace.csi[0]` (tx0, rx0 and rx1) as complex64 `(2, packets, 58)`."""
    try:
        mat = loadmat(io.BytesIO(raw))
    except Exception as e:  # scipy raises many types for bad files
        raise DataError(f"{name}: unreadable: {e}") from e
    trace = mat.get("csi_trace")
    names = getattr(getattr(trace, "dtype", None), "names", None)
    if trace is None or not names or "csi" not in names or trace.size != 1:
        raise DataError(f"{name}: no csi_trace.csi variable")
    csi = np.asarray(trace.flat[0]["csi"])
    expected = (_N_TX, _N_RX, _N_SUBCARRIERS)
    if not np.iscomplexobj(csi) or csi.ndim != 4 or csi.shape[:3] != expected:
        raise DataError(
            f"{name}: expected complex ({_N_TX}, {_N_RX}, {_N_SUBCARRIERS}, packets), "
            f"got {csi.dtype} {csi.shape}"
        )
    if csi.shape[3] == 0:
        raise DataError(f"{name}: no packets")
    if not np.isfinite(csi).all():
        raise DataError(f"{name}: contains non-finite CSI values")
    out = np.ascontiguousarray(np.transpose(csi[0], (0, 2, 1)), dtype=np.complex64)
    if not out.any(axis=2).all():
        raise DataError(f"{name}: contains an all-zero CSI packet")
    return out


def _open_cache(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        csi = map_npy(path)
    except OSError as e:
        if is_exhaustion(e):
            raise
        return None
    except (ValueError, EOFError):
        return None
    if (
        csi.ndim != 3
        or csi.shape[0] != _N_RX
        or csi.shape[2] != _N_SUBCARRIERS
        or csi.dtype != np.complex64
    ):
        return None
    return csi


def _replace_atomic(path: Path, write: Callable[[BinaryIO], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _write_cache(path: Path, csi: np.ndarray) -> None:
    _replace_atomic(path, lambda f: np.save(f, csi, allow_pickle=False))


def _read_index(path: Path) -> dict:
    """`{relative path: [size, mtime_ns, sha256]}`; empty when missing or unreadable."""
    try:
        index = json.loads(path.read_text())
    except OSError as e:
        if is_exhaustion(e):
            raise
        return {}
    except ValueError:
        return {}
    return index if isinstance(index, dict) else {}


def _load(root: Path) -> list[Recording]:
    """`_load_all`, naming fd or mapping exhaustion instead of a bare OSError."""
    try:
        return _load_all(root)
    except OSError as e:
        if not is_exhaustion(e):
            raise
        raise DataError(
            f"{root}: ran out of file descriptors, memory mappings or memory while loading: "
            f"{e}; run fewer loads or workers in parallel"
        ) from e


def _load_all(root: Path) -> list[Recording]:
    if not root.is_dir():
        raise DataError(f"dataset root not found: {root}")
    cache = root / ".cache"
    # Arrays live under the adapter version, so a changed conversion never reuses them.
    arrays = cache / f"v{CsiBenchHp.VERSION}"
    index_path = cache / "index.json"
    index = _read_index(index_path)
    changed = False
    recordings = []
    for path in sorted(root.glob(_PATTERN)):
        activity = path.parent.parent.parent.name.removeprefix("act_")
        if activity not in _ACTIVITIES:
            continue
        room = path.parent.parent.name.removeprefix("env_")
        key = path.relative_to(root).as_posix()
        st = path.stat()
        raw = None
        entry = index.get(key)
        if (
            isinstance(entry, list)
            and len(entry) == 3
            and entry[:2] == [st.st_size, st.st_mtime_ns]
            and isinstance(entry[2], str)
        ):
            sha = entry[2]
        else:
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            index[key] = [st.st_size, st.st_mtime_ns, sha]
            changed = True
        cache_path = arrays / f"{sha}.npy"
        csi = _open_cache(cache_path)
        if csi is None:
            if raw is None:
                raw = path.read_bytes()
            _write_cache(cache_path, _convert(raw, key))
            csi = _open_cache(cache_path)
            if csi is None:
                raise DataError(f"{key}: cache file unreadable: {cache_path}")
            logger.debug("converted %s", key)
        del raw
        meta = RecordingMeta(
            n_packets=csi.shape[1],
            source_sha256=sha,
            downsample_factor=None,
            sample_rate_hz=None,
        )
        recordings.append(
            Recording(
                csi=csi,
                antenna_ids=(0, 1),
                recording_id=f"{_PREFIX}/{path.relative_to(root).with_suffix('').as_posix()}",
                room=room,
                label=_ACTIVITIES[activity],
                meta=meta,
            )
        )
    if changed:
        data = json.dumps(index, sort_keys=True).encode()
        _replace_atomic(index_path, lambda f: f.write(data))
    if not recordings:
        raise DataError(f"no HP HAR recordings (.mat) in {root}")
    return sorted(recordings, key=lambda r: r.recording_id)


@register("dataset", "csi_bench_hp")
class CsiBenchHp:
    VERSION = 1
    classes: tuple[str, ...] = CLASSES

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        root: str = "data/csi_bench/Multitask/sub_Human_mat"

    def __init__(self, params: "CsiBenchHp.Params | None" = None) -> None:
        self.params = params or self.Params()

    def load(self) -> list[Recording]:
        return _load(Path(self.params.root))
