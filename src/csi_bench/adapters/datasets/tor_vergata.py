"""The own dataset: 19 saved CSI recordings in two rooms."""

import hashlib
import io
import logging
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from csi_bench.core.contracts import Recording, RecordingMeta
from csi_bench.core.errors import DataError
from csi_bench.core.registry import register

logger = logging.getLogger(__name__)

CLASSES = ("empty", "sitting", "standing", "walking")
_ACTIVITY = {
    "empty": "empty",
    "sit": "sitting",
    "sitting": "sitting",
    "standing": "standing",
    "walk": "walking",
    "walking": "walking",
}
_NAME = re.compile(r"^(r[12])_([a-z]+)(?:_(\d+))?$")
# From the csi_saver.py history: packets were kept every k-th.
_DOWNSAMPLE = {"r2_standing_2": 5, "r2_standing_1": 3}
_N_ANTENNAS = 4


def _parse(stem: str) -> tuple[str, int]:
    m = _NAME.match(stem)
    activity = _ACTIVITY.get(m.group(2)) if m else None
    if m is None or activity is None:
        raise DataError(f"unrecognised recording file name: {stem}")
    return m.group(1), CLASSES.index(activity)


def _downsample(stem: str, label: int, room: str) -> int | None:
    if stem in _DOWNSAMPLE:
        return _DOWNSAMPLE[stem]
    if room == "r1" and CLASSES[label] == "sitting":
        return 2
    return None


@register("dataset", "tor_vergata")
class TorVergata:
    VERSION = 1
    classes = CLASSES

    class Params(BaseModel):
        root: str = "data/csi_proc_python/saved_csi"

    def __init__(self, params: "TorVergata.Params | None" = None) -> None:
        self.params = params or self.Params()

    def load(self) -> list[Recording]:
        root = Path(self.params.root)
        if not root.is_dir():
            raise DataError(f"dataset root not found: {root}")
        paths = sorted(root.glob("*.npy"))
        if not paths:
            raise DataError(f"no .npy recordings in {root}")
        recordings = []
        for path in paths:
            room, label = _parse(path.stem)
            raw = path.read_bytes()
            try:
                csi = np.load(io.BytesIO(raw), allow_pickle=False)
            except (ValueError, OSError, EOFError) as e:
                raise DataError(f"{path.name}: unreadable: {e}") from e
            if csi.ndim != 3 or csi.shape[0] != _N_ANTENNAS or csi.dtype != np.complex64:
                raise DataError(
                    f"{path.name}: expected complex64 (4, packets, subcarriers), "
                    f"got {csi.dtype} {csi.shape}"
                )
            csi.setflags(write=False)
            meta = RecordingMeta(
                n_packets=csi.shape[1],
                source_sha256=hashlib.sha256(raw).hexdigest(),
                downsample_factor=_downsample(path.stem, label, room),
                sample_rate_hz=None,
            )
            recordings.append(
                Recording(
                    csi=csi,
                    antenna_ids=tuple(range(_N_ANTENNAS)),
                    recording_id=f"tor_vergata/{path.stem}",
                    room=room,
                    label=label,
                    meta=meta,
                )
            )
            logger.debug("loaded %s", path.name)
        return sorted(recordings, key=lambda r: r.recording_id)
