import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from csi_bench.adapters.datasets import csi_bench_hp
from csi_bench.adapters.datasets.csi_bench_hp import CsiBenchHp
from csi_bench.adapters.splitters.time_block import TimeBlock
from csi_bench.core.errors import DataError
from csi_bench.core.splitplan import load_split_plan
from tests.contracts.dataset_source import check_dataset_source

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/csi_bench/Multitask/sub_Human_mat"
COMMITTED = ROOT / "splits/csi_bench_hp"
FILE = "Continuous_Data_Collection-2025-5-7_20_58_45_1B31WA000383_5G_concat"


def _csi(packets: int = 12, seed: int = 0, shape: tuple | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = shape or (2, 2, 58, packets)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape) + 5


def _write(
    root: Path,
    act: str = "jumping",
    env: str = "E01",
    csi: np.ndarray | None = None,
    user: str = "U01",
    device: str = "device_HP",
    name: str = FILE,
    var: dict | None = None,
) -> Path:
    d = root / f"user_{user}" / f"act_{act}" / f"env_{env}" / device
    d.mkdir(parents=True, exist_ok=True)
    arr = _csi() if csi is None else csi
    content = var if var is not None else {"csi_trace": {"csi": arr, "mactimer": np.arange(3.0)}}
    savemat(d / f"{name}.mat", content)
    return d / f"{name}.mat"


def _load(root: Path):
    return CsiBenchHp(CsiBenchHp.Params(root=str(root))).load()


def test_conversion_matches_hand_built(tmp_path):
    csi = _csi(9)
    _write(tmp_path, csi=csi)
    (rec,) = _load(tmp_path)
    assert rec.csi.shape == (2, 9, 58)
    assert rec.csi.dtype == np.complex64
    assert rec.recording_id == f"csi_bench_hp/user_U01/act_jumping/env_E01/device_HP/{FILE}"
    assert (rec.room, rec.label, rec.antenna_ids) == ("E01", 0, (0, 1))
    assert rec.meta.n_packets == 9
    assert rec.meta.sample_rate_hz is None and rec.meta.downsample_factor is None
    for rx in range(2):
        for p in range(9):
            np.testing.assert_array_equal(rec.csi[rx, p], csi[0, rx, :, p].astype(np.complex64))
    with pytest.raises(ValueError):
        rec.csi[0, 0, 0] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    ("act", "label"),
    [
        ("jumping", 0),
        ("running", 1),
        ("seated-breathing", 2),
        ("walking-case1", 3),
        ("walking-case2", 3),
        ("walking-case3", 3),
        ("walking-case4", 3),
        ("wavinghand", 4),
    ],
)
def test_labels(tmp_path, act, label):
    _write(tmp_path, act=act, env="E27")
    (rec,) = _load(tmp_path)
    assert (rec.label, rec.room) == (label, "E27")
    assert CsiBenchHp.classes[label] == {
        "seated-breathing": "seated_breathing",
        "wavinghand": "waving_hand",
    }.get(act, act.split("-")[0])


def test_out_of_scope_skipped(tmp_path):
    _write(tmp_path, act="walking-case3", env="E02")
    bad = _csi(shape=(1, 4, 14, 5))  # would fail conversion if it were read
    _write(tmp_path, act="localization-1", csi=bad)
    _write(tmp_path, device="device_AmazonPlug", csi=bad)
    _write(tmp_path, device="device_HP_1B31WA000383", csi=bad)
    (tmp_path / "user_U01" / "act_walking-case3" / "env_E02" / "device_HP" / "x.h5").write_bytes(
        b"x"
    )
    source = CsiBenchHp(CsiBenchHp.Params(root=str(tmp_path)))
    check_dataset_source(source)
    (rec,) = source.load()
    assert (rec.label, rec.room) == (3, "E02")


def test_sorted_ids(tmp_path):
    for user in ("U03", "U01", "U02"):
        _write(tmp_path, user=user)
    ids = [r.recording_id for r in _load(tmp_path)]
    assert ids == sorted(ids) and len(ids) == 3


def _nan() -> np.ndarray:
    csi = _csi()
    csi[1, 0, 3, 4] = np.nan  # tx1 too: the whole file must be finite
    return csi


def _zero_packet() -> np.ndarray:
    csi = _csi()
    csi[0, 1, :, 5] = 0
    return csi


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ({"other": np.ones(3)}, "no csi_trace.csi"),
        ({"csi_trace": {"mactimer": np.ones(3)}}, "no csi_trace.csi"),
        ({"csi_trace": np.ones(3)}, "no csi_trace.csi"),
        ({"csi_trace": {"csi": _csi().real}}, "expected complex"),
        ({"csi_trace": {"csi": _csi(shape=(2, 1, 58, 12))}}, "expected complex"),
        ({"csi_trace": {"csi": _csi(shape=(2, 2, 56, 12))}}, "expected complex"),
        ({"csi_trace": {"csi": _csi(shape=(2, 2, 58))}}, "expected complex"),
        ({"csi_trace": {"csi": _nan()}}, "contains non-finite"),
        ({"csi_trace": {"csi": _zero_packet()}}, "contains an all-zero CSI packet"),
    ],
)
def test_bad_content(tmp_path, content, error):
    _write(tmp_path, var=content)
    with pytest.raises(DataError, match=f"{FILE}.mat: {error}"):
        _load(tmp_path)


def test_unreadable_file(tmp_path):
    path = _write(tmp_path)
    path.write_bytes(b"not a mat file")
    with pytest.raises(DataError, match=f"{FILE}.mat: unreadable"):
        _load(tmp_path)


def test_missing_root_and_no_files(tmp_path):
    with pytest.raises(DataError, match="not found"):
        _load(tmp_path / "nope")
    with pytest.raises(DataError, match="no HP HAR"):
        _load(tmp_path)
    _write(tmp_path, act="localization-1")
    _write(tmp_path, device="device_AmazonPlug")
    with pytest.raises(DataError, match="no HP HAR"):
        _load(tmp_path)


def test_params_forbid_extra():
    with pytest.raises(ValueError):
        CsiBenchHp.Params.model_validate({"bogus": 1})
    assert CsiBenchHp.Params().root == "data/csi_bench/Multitask/sub_Human_mat"


def test_cache_hit_and_reconvert_on_change(tmp_path, monkeypatch):
    path = _write(tmp_path, csi=_csi(12, seed=1))
    first = _load(tmp_path)[0]
    cached = list((tmp_path / ".cache").rglob("*.npy"))
    assert len(cached) == 1
    assert cached[0].parent.name == f"v{CsiBenchHp.VERSION}"
    assert cached[0].stem == first.meta.source_sha256
    assert not list((tmp_path / ".cache").rglob("*.tmp"))

    def boom(*a, **k):
        raise AssertionError("cache miss: .mat read again")

    monkeypatch.setattr(csi_bench_hp, "loadmat", boom)
    monkeypatch.setattr(Path, "read_bytes", boom)
    second = _load(tmp_path)[0]
    assert not second.csi.flags.owndata  # mapped from the cache file, not read into memory
    assert not second.csi.flags.writeable
    assert second.meta.source_sha256 == first.meta.source_sha256
    np.testing.assert_array_equal(first.csi, second.csi)
    monkeypatch.undo()

    # same size, new content and mtime: re-hashed and re-converted
    mtime = path.stat().st_mtime_ns
    _write(tmp_path, csi=_csi(12, seed=2))
    os.utime(path, ns=(mtime + 1, mtime + 1))
    third = _load(tmp_path)[0]
    assert third.meta.source_sha256 != first.meta.source_sha256
    assert not np.array_equal(np.asarray(first.csi), np.asarray(third.csi))
    _write(tmp_path, csi=_csi(20, seed=3))
    assert _load(tmp_path)[0].csi.shape == (2, 20, 58)
    assert len(list((tmp_path / ".cache").rglob("*.npy"))) == 3


def test_corrupt_index_and_cache(tmp_path):
    _write(tmp_path)
    first = _load(tmp_path)[0]
    expected, sha = np.array(first.csi), first.meta.source_sha256
    del first  # truncating a file that is still mapped would crash the process
    (tmp_path / ".cache" / "index.json").write_text("{not json")
    (tmp_path / ".cache" / f"v{CsiBenchHp.VERSION}" / f"{sha}.npy").write_bytes(b"junk")
    again = _load(tmp_path)[0]
    assert again.meta.source_sha256 == sha
    np.testing.assert_array_equal(expected, again.csi)


def test_reconvert_when_index_hits_but_array_missing(tmp_path, monkeypatch):
    # A VERSION bump keeps index.json valid but has no array yet: the .mat is read again.
    _write(tmp_path)
    first = _load(tmp_path)[0]
    expected, sha = np.array(first.csi), first.meta.source_sha256
    index = (tmp_path / ".cache" / "index.json").read_text()
    monkeypatch.setattr(CsiBenchHp, "VERSION", CsiBenchHp.VERSION + 1)
    again = _load(tmp_path)[0]
    assert again.meta.source_sha256 == sha
    np.testing.assert_array_equal(expected, again.csi)
    assert (tmp_path / ".cache" / f"v{CsiBenchHp.VERSION}" / f"{sha}.npy").is_file()
    assert (tmp_path / ".cache" / "index.json").read_text() == index  # index hit, no re-hash


@pytest.mark.skipif(not REAL.is_dir(), reason="dataset not present")
def test_real_dataset():
    source = CsiBenchHp(CsiBenchHp.Params(root=str(REAL)))
    check_dataset_source(source)
    recs = source.load()
    assert len(recs) == 438
    rooms = Counter(r.room for r in recs)
    assert rooms == {
        "E01": 126,
        "E02": 48,
        "E03": 56,
        "E04": 57,
        "E05": 57,
        "E06": 40,
        "E27": 54,
    }
    for room in rooms:
        assert {r.label for r in recs if r.room == room} == set(range(5)), room
    assert all(r.csi.shape[0] == 2 and r.csi.shape[2] == 58 for r in recs)
    assert min(r.meta.n_packets for r in recs) == 6843
    assert max(r.meta.n_packets for r in recs) == 32705
    serials = Counter(r.recording_id.rsplit("_", 3)[1] for r in recs)
    assert serials == {"1B31WA000383": 140, "1B31WA000430": 150, "1B31WA000514": 148}
    first = recs[0]
    rel = first.recording_id.split("/", 1)[1]
    raw = loadmat(REAL / f"{rel}.mat")["csi_trace"][0, 0]["csi"]
    np.testing.assert_array_equal(first.csi, raw[0].transpose(0, 2, 1).astype(np.complex64))
    plan = TimeBlock().split(recs, "csi_bench_hp")
    committed = COMMITTED / f"{plan.split_hash}.json"
    assert committed.is_file(), f"generate with write_split_plan: {committed}"
    assert load_split_plan(committed, recs) == plan
    long = TimeBlock(TimeBlock.Params(gap_packets=610)).split(recs, "csi_bench_hp")
    assert long.split_hash == "d6b97e4f25b3"  # the Mac long-window plan (commit 05237c4)
    assert load_split_plan(COMMITTED / "d6b97e4f25b3.json", recs) == long
