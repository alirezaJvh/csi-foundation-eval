import gc
import os

import numpy as np
import pytest

from csi_bench.core.npy_map import is_exhaustion, map_npy


def _fds() -> int:
    return len(os.listdir("/dev/fd"))


def _mapped(path) -> bool:
    maps = "/proc/self/maps"
    if not os.path.exists(maps):
        pytest.skip("needs /proc/self/maps")
    with open(maps) as f:
        return str(path) in f.read()


@pytest.mark.parametrize(
    "array",
    [
        (np.arange(24, dtype=np.float32) * 1.5).reshape(2, 3, 4),
        (np.arange(30) + 1j * np.arange(30)).astype(np.complex64).reshape(2, 5, 3),
        np.arange(7, dtype=np.int64),
        np.float64(3.25),
        np.zeros((0, 4), dtype=np.float32),
    ],
)
def test_values_equal_np_load(tmp_path, array):
    path = tmp_path / "a.npy"
    np.save(path, array)
    mapped = map_npy(path)
    expected = np.load(path)
    assert mapped.dtype == expected.dtype
    assert mapped.shape == expected.shape
    np.testing.assert_array_equal(mapped, expected)


def test_read_only(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.ones((3, 4), dtype=np.complex64))
    mapped = map_npy(path)
    assert not mapped.flags.writeable
    with pytest.raises(ValueError):
        mapped[0, 0] = 2
    with pytest.raises(ValueError):
        mapped.flags.writeable = True


def test_no_fd_kept(tmp_path):
    paths = []
    for i in range(20):
        paths.append(tmp_path / f"{i}.npy")
        np.save(paths[-1], np.full((2, 8, 3), i, dtype=np.complex64))
    before = _fds()
    arrays = [map_npy(p) for p in paths]
    assert _fds() == before
    assert [int(a[0, 0, 0].real) for a in arrays] == list(range(20))


def test_unmapped_after_release(tmp_path):
    path = tmp_path / "released.npy"
    np.save(path, np.arange(4096, dtype=np.float64))
    mapped = map_npy(path)
    view = mapped[10:20]
    del mapped
    gc.collect()
    assert _mapped(path)  # a view keeps the mapping alive
    np.testing.assert_array_equal(view, np.arange(10, 20, dtype=np.float64))
    del view
    gc.collect()
    assert not _mapped(path)


def test_bad_files_raise_value_error(tmp_path):
    bad = tmp_path / "bad.npy"
    bad.write_bytes(b"not an npy file at all")
    with pytest.raises(ValueError):
        map_npy(bad)
    empty = tmp_path / "empty.npy"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        map_npy(empty)
    truncated = tmp_path / "truncated.npy"
    np.save(truncated, np.arange(100, dtype=np.float64))
    truncated.write_bytes(truncated.read_bytes()[:-8])
    with pytest.raises(ValueError, match="truncated"):
        map_npy(truncated)
    fortran = tmp_path / "fortran.npy"
    np.save(fortran, np.asfortranarray(np.arange(6.0).reshape(2, 3)))
    with pytest.raises(ValueError, match="Fortran"):
        map_npy(fortran)
    objects = tmp_path / "objects.npy"
    np.save(objects, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="object"):
        map_npy(objects)


def test_missing_file_is_os_error(tmp_path):
    with pytest.raises(OSError) as info:
        map_npy(tmp_path / "missing.npy")
    assert not is_exhaustion(info.value)
