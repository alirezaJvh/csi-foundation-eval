import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from csi_bench.adapters.datasets import padova
from csi_bench.adapters.datasets.padova import Padova
from csi_bench.adapters.splitters.time_block import TimeBlock
from csi_bench.core.errors import DataError
from csi_bench.core.splitplan import load_split_plan
from tests.contracts.dataset_source import check_dataset_source

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/padova"
COMMITTED = ROOT / "splits/padova"
DROPPED = [0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, 203, 231]
DROPPED += [251, 252, 253, 254, 255]


def _rows(n_rows: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_rows, 256)) + 1j * rng.normal(size=(n_rows, 256)) + 5


def _write(root: Path, name: str, rows: np.ndarray | None = None, var: str = "csi_buff"):
    d = root / name.split("_")[0]
    d.mkdir(parents=True, exist_ok=True)
    savemat(d / f"{name}.mat", {var: _rows(16) if rows is None else rows})


def _load(root: Path):
    return Padova(Padova.Params(root=str(root))).load()


@pytest.mark.skipif(not (REAL / "S1a").is_dir(), reason="dataset not present")
def test_real_dataset():
    source = Padova(Padova.Params(root=str(REAL)))
    check_dataset_source(source)
    recs = source.load()
    assert len(recs) == 105
    assert all(r.csi.shape[0] == 4 and r.csi.shape[2] == 234 for r in recs)
    assert all(r.meta.sample_rate_hz == 173.0 for r in recs)
    assert {r.room for r in recs} == {"bedroom", "living_room", "laboratory"}
    assert {r.label for r in recs} == set(range(8))
    by_id = {r.recording_id: r for r in recs}
    c = by_id["padova/S1a/S1a_C"]
    assert c.csi.shape == (4, 20397, 234)
    assert c.label == 6
    assert by_id["padova/S7a/S7a_L2"].label == by_id["padova/S7a/S7a_L1"].label
    for r in recs:
        expected = {"S6": "living_room", "S7": "laboratory"}.get(r.recording_id[7:9], "bedroom")
        assert r.room == expected, r.recording_id
    for room in ("bedroom", "living_room", "laboratory"):
        assert {r.label for r in recs if r.room == room} == set(range(8)), room
    plan = TimeBlock().split(recs, "padova")
    committed = COMMITTED / f"{plan.split_hash}.json"
    assert committed.is_file(), f"generate with write_split_plan: {committed}"
    assert load_split_plan(committed, recs) == plan


def test_conversion_matches_hand_built(tmp_path):
    rows = _rows(4 * 7 + 2)  # 2 trailing rows are dropped
    _write(tmp_path, "S1a_W", rows)
    (rec,) = _load(tmp_path)
    assert rec.csi.shape == (4, 7, 234)
    assert rec.meta.n_packets == 7
    assert rec.label == 1
    assert rec.room == "bedroom"
    assert rec.recording_id == "padova/S1a/S1a_W"
    assert rec.antenna_ids == (0, 1, 2, 3)
    assert rec.meta.sample_rate_hz == 173.0
    assert rec.meta.downsample_factor is None
    keep = [i for i in range(256) if i not in DROPPED]
    for a in range(4):
        for p in range(7):
            shifted = np.fft.fftshift(rows[4 * p + a])
            np.testing.assert_allclose(rec.csi[a, p], shifted[keep].astype(np.complex64))
    with pytest.raises(ValueError):
        rec.csi[0, 0, 0] = 1  # type: ignore[index]


def test_rooms_labels_and_trials(tmp_path):
    for n in ["S1a_E", "S6a_R", "S7a_L", "S7a_L2", "S2b_J1", "S3a_H", "S4a_S"]:
        _write(tmp_path, n)
    recs = {r.recording_id.split("/")[2]: r for r in _load(tmp_path)}
    assert recs["S1a_E"].label == 0
    assert (recs["S6a_R"].room, recs["S6a_R"].label) == ("living_room", 2)
    assert (recs["S7a_L"].room, recs["S7a_L"].label) == ("laboratory", 4)
    assert recs["S7a_L2"].label == 4
    assert recs["S7a_L2"].recording_id == "padova/S7a/S7a_L2"
    assert recs["S2b_J1"].label == 3
    assert recs["S3a_H"].label == 7
    assert recs["S4a_S"].label == 5


@pytest.mark.parametrize(
    ("folder", "name", "error"),
    [
        ("S1a", "S1a_X", "unrecognised"),
        ("S1a", "foo", "unrecognised"),
        ("S1a", "S1a_", "unrecognised"),
        ("S1a", "S2a_E", "unrecognised"),  # file in another set's folder
        ("S9a", "S9a_E", "unknown recording set"),
    ],
)
def test_bad_name(tmp_path, folder, name, error):
    d = tmp_path / folder
    d.mkdir()
    savemat(d / f"{name}.mat", {"csi_buff": _rows(8)})
    with pytest.raises(DataError, match=f"{error}.*{name}"):
        _load(tmp_path)


def test_zero_row(tmp_path):
    rows = _rows(8)
    rows[5] = 0
    _write(tmp_path, "S1a_E", rows)
    with pytest.raises(DataError, match="S1a_E.mat"):
        _load(tmp_path)


def test_missing_variable_and_bad_width(tmp_path):
    _write(tmp_path, "S1a_E", var="other")
    with pytest.raises(DataError, match="S1a_E.mat"):
        _load(tmp_path)
    _write(tmp_path, "S1a_E", rows=_rows(8)[:, :100])
    with pytest.raises(DataError, match="S1a_E.mat"):
        _load(tmp_path)


def test_real_valued_and_too_few_rows(tmp_path):
    _write(tmp_path, "S1a_E", rows=_rows(8).real)
    with pytest.raises(DataError, match="S1a_E.mat: expected complex"):
        _load(tmp_path)
    _write(tmp_path, "S1a_E", rows=_rows(3))
    with pytest.raises(DataError, match="S1a_E.mat: fewer than 4"):
        _load(tmp_path)


def test_unreadable_file(tmp_path):
    (tmp_path / "S1a").mkdir()
    (tmp_path / "S1a" / "S1a_E.mat").write_bytes(b"not a mat file")
    with pytest.raises(DataError, match="S1a_E.mat"):
        _load(tmp_path)


def test_empty_and_missing_root(tmp_path):
    with pytest.raises(DataError, match="no .mat"):
        _load(tmp_path)
    with pytest.raises(DataError):
        _load(tmp_path / "nope")


def test_params_forbid_extra():
    with pytest.raises(ValueError):
        Padova.Params.model_validate({"bogus": 1})
    assert Padova.Params().root == "data/padova"


def test_cache_hit_and_invalidation(tmp_path, monkeypatch):
    _write(tmp_path, "S1a_E", _rows(16, seed=1))
    first = _load(tmp_path)[0]
    cached = list((tmp_path / ".cache").rglob("*.npy"))
    assert len(cached) == 1
    assert cached[0].parent.name == f"v{Padova.VERSION}"
    assert cached[0].stem == first.meta.source_sha256
    assert not list((tmp_path / ".cache").rglob("*.tmp"))

    def boom(*a, **k):
        raise AssertionError("cache miss: .mat read again")

    monkeypatch.setattr(padova, "loadmat", boom)
    monkeypatch.setattr(Path, "read_bytes", boom)  # an unchanged file is not even hashed
    second = _load(tmp_path)[0]
    assert isinstance(second.csi, np.memmap)
    assert not second.csi.flags.writeable
    assert second.meta.source_sha256 == first.meta.source_sha256
    np.testing.assert_array_equal(first.csi, second.csi)
    monkeypatch.undo()

    _write(tmp_path, "S1a_E", _rows(20, seed=2))
    third = _load(tmp_path)[0]
    assert third.meta.source_sha256 != first.meta.source_sha256
    assert third.csi.shape == (4, 5, 234)
    assert len(list((tmp_path / ".cache").rglob("*.npy"))) == 2


def test_same_size_rewrite_is_rehashed(tmp_path):
    _write(tmp_path, "S1a_E", _rows(16, seed=1))
    first = _load(tmp_path)[0]
    path = tmp_path / "S1a" / "S1a_E.mat"
    size, mtime = path.stat().st_size, path.stat().st_mtime_ns
    _write(tmp_path, "S1a_E", _rows(16, seed=2))
    os.utime(path, ns=(mtime + 1, mtime + 1))
    assert path.stat().st_size == size
    second = _load(tmp_path)[0]
    assert second.meta.source_sha256 != first.meta.source_sha256
    assert not np.array_equal(np.asarray(first.csi), np.asarray(second.csi))


def test_corrupt_index_is_ignored(tmp_path):
    _write(tmp_path, "S1a_E")
    sha = _load(tmp_path)[0].meta.source_sha256
    (tmp_path / ".cache" / "index.json").write_text("{not json")
    assert _load(tmp_path)[0].meta.source_sha256 == sha


def test_corrupt_cache_is_rebuilt(tmp_path):
    _write(tmp_path, "S1a_E")
    first = _load(tmp_path)[0]
    expected, sha = np.array(first.csi), first.meta.source_sha256
    del first  # truncating a file that is still mapped would crash the process
    (tmp_path / ".cache" / f"v{Padova.VERSION}" / f"{sha}.npy").write_bytes(b"junk")
    again = _load(tmp_path)[0]
    np.testing.assert_array_equal(expected, again.csi)
