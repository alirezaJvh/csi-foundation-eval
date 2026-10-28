"""The public Padova 80 MHz dataset (SHARP, Meneghello et al., IEEE TMC 2022)."""

import hashlib
import io
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.io import loadmat

from csi_bench.core.contracts import Recording, RecordingMeta
from csi_bench.core.errors import DataError
from csi_bench.core.registry import register

logger = logging.getLogger(__name__)

CLASSES = (
    "empty",
    "walking",
    "running",
    "jumping",
    "sitting",
    "standing",
    "sit_stand",
    "arm_gym",
)
_LETTERS = "EWRJLSCH"  # same order as CLASSES
_NAME = re.compile(r"^(S(\d+)[a-z])_([A-Z])(\d*)$")
_ROOMS = {1: "bedroom", 2: "bedroom", 3: "bedroom", 4: "bedroom", 5: "bedroom"}
_ROOMS.update({6: "living_room", 7: "laboratory"})
_N_ANTENNAS = 4
_RAW_WIDTH = 256
# The same null, DC and pilot indices the own data drops (csi_decoder.py), after the shift.
_DROPPED = [0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, 203, 231] + [
    251,
    252,
    253,
    254,
    255,
]
_SAMPLE_RATE_HZ = 173.0


def _parse(path: Path) -> tuple[str, str, int]:
    """Return (set name, room, label) for `<set>/<set>_<activity>[n].mat`."""
    m = _NAME.match(path.stem)
    if m is None or m.group(3) not in _LETTERS or path.parent.name != m.group(1):
        raise DataError(f"unrecognised recording file name: {path.parent.name}/{path.name}")
    room = _ROOMS.get(int(m.group(2)))
    if room is None:
        raise DataError(f"unknown recording set: {path.parent.name}/{path.name}")
    return m.group(1), room, _LETTERS.index(m.group(3))


def _convert(raw: bytes, name: str) -> np.ndarray:
    try:
        mat = loadmat(io.BytesIO(raw))
    except Exception as e:  # scipy raises many types for bad files
        raise DataError(f"{name}: unreadable: {e}") from e
    if "csi_buff" not in mat:
        raise DataError(f"{name}: no csi_buff variable")
    rows = np.asarray(mat["csi_buff"])
    if rows.ndim != 2 or rows.shape[1] != _RAW_WIDTH or not np.iscomplexobj(rows):
        raise DataError(
            f"{name}: expected complex (rows, {_RAW_WIDTH}), got {rows.dtype} {rows.shape}"
        )
    rows = rows[: rows.shape[0] // _N_ANTENNAS * _N_ANTENNAS]
    if rows.shape[0] == 0:
        raise DataError(f"{name}: fewer than {_N_ANTENNAS} CSI rows")
    if not rows.any(axis=1).all():
        raise DataError(f"{name}: contains an all-zero CSI row")
    n = rows.shape[0] // _N_ANTENNAS
    keep = _RAW_WIDTH - len(_DROPPED)
    out = np.empty((_N_ANTENNAS, n, keep), dtype=np.complex64)
    for a in range(_N_ANTENNAS):
        shifted = np.fft.fftshift(rows[a::_N_ANTENNAS], axes=1)
        out[a] = np.delete(shifted, _DROPPED, axis=1)
    return out


def _open_cache(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        csi = np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, OSError, EOFError):
        return None
    if csi.ndim != 3 or csi.shape[0] != _N_ANTENNAS or csi.dtype != np.complex64:
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
    except (OSError, ValueError):
        return {}
    return index if isinstance(index, dict) else {}


@register("dataset", "padova")
class Padova:
    VERSION = 1
    classes = CLASSES

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        root: str = "data/padova"

    def __init__(self, params: "Padova.Params | None" = None) -> None:
        self.params = params or self.Params()

    def load(self) -> list[Recording]:
        root = Path(self.params.root)
        if not root.is_dir():
            raise DataError(f"dataset root not found: {root}")
        paths = sorted(root.glob("*/*.mat"))
        if not paths:
            raise DataError(f"no .mat recordings in {root}")
        cache = root / ".cache"
        # Arrays live under the adapter version, so a changed conversion never reuses them.
        arrays = cache / f"v{self.VERSION}"
        index_path = cache / "index.json"
        index = _read_index(index_path)
        changed = False
        recordings = []
        for path in paths:
            set_name, room, label = _parse(path)
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
                _write_cache(cache_path, _convert(raw, f"{set_name}/{path.name}"))
                csi = _open_cache(cache_path)
                if csi is None:
                    raise DataError(f"{path.name}: cache file unreadable: {cache_path}")
                logger.debug("converted %s", path.name)
            del raw
            meta = RecordingMeta(
                n_packets=csi.shape[1],
                source_sha256=sha,
                downsample_factor=None,
                sample_rate_hz=_SAMPLE_RATE_HZ,
            )
            recordings.append(
                Recording(
                    csi=csi,
                    antenna_ids=tuple(range(_N_ANTENNAS)),
                    recording_id=f"padova/{set_name}/{path.stem}",
                    room=room,
                    label=label,
                    meta=meta,
                )
            )
        if changed:
            data = json.dumps(index, sort_keys=True).encode()
            _replace_atomic(index_path, lambda f: f.write(data))
        return sorted(recordings, key=lambda r: r.recording_id)
