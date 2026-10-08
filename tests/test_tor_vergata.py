import sys
from pathlib import Path

import numpy as np
import pytest

from csi_bench.adapters.datasets.tor_vergata import TorVergata
from csi_bench.core import registry
from csi_bench.core.errors import DataError
from csi_bench.runner.discover import discover
from tests.contracts.dataset_source import check_dataset_source

REAL = Path(__file__).resolve().parents[1] / "data/csi_proc_python/saved_csi"


def _write(root: Path, name: str, shape=(4, 5, 6), dtype=np.complex64):
    np.save(root / f"{name}.npy", np.zeros(shape, dtype=dtype))


def _load(root: Path):
    return TorVergata(TorVergata.Params(root=str(root))).load()


@pytest.mark.skipif(not REAL.is_dir(), reason="dataset not present")
def test_real_dataset():
    source = TorVergata(TorVergata.Params(root=str(REAL)))
    check_dataset_source(source)
    recs = source.load()
    assert len(recs) == 19
    assert all(r.csi.shape[0] == 4 and r.csi.shape[2] == 234 for r in recs)
    assert all(r.antenna_ids == (0, 1, 2, 3) for r in recs)
    assert {r.room for r in recs} == {"r1", "r2"}
    by_id = {r.recording_id: r for r in recs}
    assert by_id["tor_vergata/r1_empty"].label == 0
    assert by_id["tor_vergata/r2_standing_2"].meta.downsample_factor == 5
    assert by_id["tor_vergata/r2_standing_1"].meta.downsample_factor == 3
    expected = {"sitting": 1, "standing": 2, "walking": 3}
    for r in recs:
        stem = r.recording_id.split("/")[1]
        activity = stem.split("_")[1]
        if activity in ("sit", "sitting", "standing", "walk", "walking"):
            key = {"sit": "sitting", "walk": "walking"}.get(activity, activity)
            assert r.label == expected[key], stem
        if stem.startswith("r1_sitting"):
            assert r.meta.downsample_factor == 2, stem


def test_name_variants_and_downsample(tmp_path):
    names = [
        "r1_empty",
        "r2_sit_1",
        "r2_walk_2",
        "r1_sitting_3",
        "r2_standing_1",
        "r2_standing_2",
        "r1_standing_1",
    ]
    for n in names:
        _write(tmp_path, n)
    recs = {r.recording_id.split("/")[1]: r for r in _load(tmp_path)}
    assert (recs["r1_empty"].room, recs["r1_empty"].label) == ("r1", 0)
    assert (recs["r2_sit_1"].room, recs["r2_sit_1"].label) == ("r2", 1)
    assert (recs["r2_walk_2"].room, recs["r2_walk_2"].label) == ("r2", 3)
    assert recs["r1_standing_1"].label == 2
    assert recs["r2_standing_2"].meta.downsample_factor == 5
    assert recs["r2_standing_1"].meta.downsample_factor == 3
    assert recs["r1_sitting_3"].meta.downsample_factor == 2
    assert recs["r1_empty"].meta.downsample_factor is None
    assert recs["r1_empty"].meta.sample_rate_hz is None
    assert recs["r2_sit_1"].meta.downsample_factor is None
    with pytest.raises(AttributeError):
        recs["r1_empty"].room = "r2"  # type: ignore[misc]
    with pytest.raises(ValueError):
        recs["r1_empty"].csi[0, 0, 0] = 1


def test_bad_name(tmp_path):
    _write(tmp_path, "r3_dance_1")
    with pytest.raises(DataError, match="r3_dance_1"):
        _load(tmp_path)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [((2, 5, 6), np.complex64), ((4, 5), np.complex64), ((4, 5, 6), np.float32)],
)
def test_bad_shape(tmp_path, shape, dtype):
    _write(tmp_path, "r1_empty", shape, dtype)
    with pytest.raises(DataError, match="r1_empty"):
        _load(tmp_path)


def test_unreadable_file(tmp_path):
    (tmp_path / "r1_empty.npy").write_bytes(b"not a npy file")
    with pytest.raises(DataError, match="r1_empty"):
        _load(tmp_path)


def test_empty_root(tmp_path):
    with pytest.raises(DataError, match="no .npy"):
        _load(tmp_path)


def test_missing_root(tmp_path):
    with pytest.raises(DataError):
        _load(tmp_path / "nope")


@pytest.fixture
def clean_registry():
    registry.reset()
    yield
    registry.reset()


def test_registered_via_discovery(monkeypatch, clean_registry):
    # a fresh import is needed for registration to run again after reset
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters.datasets")]:
        monkeypatch.delitem(sys.modules, name)
    discover()
    assert registry.get("dataset", "tor_vergata").__name__ == "TorVergata"
