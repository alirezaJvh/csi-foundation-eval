"""Map a `.npy` file read-only without keeping a file descriptor open (Story 6.5).

`np.load(mmap_mode="r")` goes through Python's `mmap`, which keeps a duplicated descriptor
per array for its whole life (3.12 has no `trackfd`), so a dataset of a few hundred cached
recordings holds a few hundred fds and parallel sweeps exhaust the limit. Here the file is
mapped through libc `mmap`, the descriptor is closed at once and the mapping is released
with `munmap` when the last array viewing it is garbage collected.
"""

import ctypes
import errno
import mmap
import os
import weakref
from pathlib import Path

import numpy as np

# errnos that mean the process or system ran out of descriptors or mappings, not a bad file
EXHAUSTION_ERRNOS = frozenset({errno.EMFILE, errno.ENFILE, errno.ENOMEM})

_libc = ctypes.CDLL(None, use_errno=True)
_mmap = _libc.mmap
_mmap.restype = ctypes.c_void_p
_mmap.argtypes = (
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int64,
)
_munmap = _libc.munmap
_munmap.restype = ctypes.c_int
_munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
_MAP_FAILED = ctypes.c_void_p(-1).value


def is_exhaustion(error: OSError) -> bool:
    """True when `error` reports fd or mapping exhaustion (EMFILE, ENFILE, ENOMEM)."""
    return error.errno in EXHAUSTION_ERRNOS


def _read_header(f) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(f)
    raise ValueError(f"unsupported .npy format version {version}")


def map_npy(path: str | Path) -> np.ndarray:
    """The array stored at `path`, mapped read-only; no fd stays open after the call.

    Raises ValueError for a malformed, truncated, Fortran-order or object-dtype file and
    OSError when the file cannot be opened or mapped (check `is_exhaustion`).
    """
    with open(path, "rb") as f:
        shape, fortran_order, dtype = _read_header(f)
        if fortran_order:
            raise ValueError(f"{path}: Fortran-order arrays are not supported")
        if dtype.hasobject:
            raise ValueError(f"{path}: object arrays are not supported")
        offset = f.tell()
        size = os.fstat(f.fileno()).st_size
        count = int(np.prod(shape, dtype=np.int64))
        if offset + count * dtype.itemsize > size:
            raise ValueError(f"{path}: truncated: header needs {count} items of {dtype}")
        address = _mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, f.fileno(), 0)
        if address is None or address == _MAP_FAILED:
            code = ctypes.get_errno()
            raise OSError(code, f"mmap failed: {os.strerror(code)}", str(path))
    # The fd is closed; the mapping stays valid until munmap.
    owner = (ctypes.c_char * size).from_address(address)
    weakref.finalize(owner, _munmap, address, size)
    view = memoryview(owner).cast("B").toreadonly()
    return np.frombuffer(view, dtype=dtype, count=count, offset=offset).reshape(shape)
