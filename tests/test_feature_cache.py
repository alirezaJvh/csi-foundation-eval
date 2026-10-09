import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from csi_bench.adapters.storage.feature_cache import NpzFeatureCache
from csi_bench.config.experiment import ExperimentConfig, feature_cache_key
from csi_bench.core import registry
from csi_bench.core.contracts import FeatureMatrix
from csi_bench.runner.discover import discover

BASE = {
    "dataset": "tor_vergata",
    "selection": {"rooms": ["r1"]},
    "preprocessing": [{"antennas": [0, 1]}, {"global_rescale": {}}],
    "window_len": 40,
    "stride": 1,
    "extractors": [{"name": "fake_ext", "params": {"scale": 1}}],
    "head": {"name": "logreg"},
    "seeds": [0],
    "role": "comparison",
}
FITTED = [{"ids": [0, 1]}, {"mean_amplitude": 1.5}]
IDS = ["a@0", "a@1", "b@0"]


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    discover()

    @registry.register("extractor", "fake_ext")
    class FakeExt:
        VERSION = 1
        granularity = "per_window"
        requires: dict = {}
        min_window_len = 1

        class Params(BaseModel):
            model_config = ConfigDict(extra="forbid")
            scale: int = 1

    @registry.register("extractor", "fake_rec")
    class FakeRec(FakeExt):
        granularity = "per_recording"

    yield
    registry.reset()


def matrix(n=3, d=2):
    values = np.arange(n * d, dtype=np.float32).reshape(n, d)
    return FeatureMatrix(values=values, feature_names=tuple(f"f{i}" for i in range(d)))


def key(changes=None, split_hash="s", fitted=None, extractor=None):
    raw = {**BASE, **(changes or {})}
    config = ExperimentConfig.model_validate(raw)
    return feature_cache_key(config, config.extractors[0], split_hash, fitted or FITTED)


def test_roundtrip(tmp_path):
    cache = NpzFeatureCache(tmp_path / "c")
    assert cache.get("k", IDS) is None
    cache.put("k", matrix(), IDS)
    got = cache.get("k", IDS)
    assert got is not None
    assert got.feature_names == ("f0", "f1")
    np.testing.assert_array_equal(got.values, matrix().values)
    assert got.values.dtype == np.float32
    assert cache.get("k") is not None


def test_stale_windows_is_a_miss_and_overwritable(tmp_path, caplog):
    cache = NpzFeatureCache(tmp_path)
    cache.put("k", matrix(), IDS)
    with caplog.at_level(logging.WARNING):
        assert cache.get("k", ["a@0", "a@1", "c@0"]) is None
    assert "different windows" in caplog.text
    cache.put("k", matrix(), ["a@0", "a@1", "c@0"])
    assert cache.get("k", ["a@0", "a@1", "c@0"]) is not None


def test_corrupt_entry_is_a_miss(tmp_path, caplog):
    cache = NpzFeatureCache(tmp_path)
    cache.put("k", matrix(), IDS)
    path = tmp_path / "k.npz"
    path.write_bytes(path.read_bytes()[:40])
    with caplog.at_level(logging.WARNING):
        assert cache.get("k", IDS) is None
    assert "unreadable" in caplog.text
    cache.put("k", matrix(), IDS)
    assert cache.get("k", IDS) is not None


def test_inconsistent_shapes_is_a_miss(tmp_path, caplog):
    cache = NpzFeatureCache(tmp_path)
    np.savez(
        tmp_path / "k.npz",
        values=np.zeros((2, 2), dtype=np.float32),
        feature_names=np.array(["f0", "f1"], dtype=str),
        window_ids=np.array(IDS, dtype=str),
    )
    with caplog.at_level(logging.WARNING):
        assert cache.get("k", IDS) is None
    assert "unreadable" in caplog.text
    cache.put("k", matrix(), IDS)
    assert cache.get("k", IDS) is not None


def test_failed_write_leaves_nothing(tmp_path, monkeypatch):
    cache = NpzFeatureCache(tmp_path)

    def boom(*args):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        cache.put("k", matrix(), IDS)
    assert list(Path(tmp_path).iterdir()) == []


def test_key_is_stable():
    assert key() == key()
    # head, seeds and role never enter the key
    assert key() == key({"head": {"name": "knn"}, "seeds": [3], "role": "reference"})


@pytest.mark.parametrize(
    "changes",
    [
        {"selection": {"rooms": ["r2"]}},
        {"selection": {"rooms": ["r1", "r2"]}},
        {"window_len": 41},
        {"stride": 2},
        {"preprocessing": [{"antennas": [0, 1]}]},
        {"preprocessing": [{"global_rescale": {}}, {"antennas": [0, 1]}]},
        {"preprocessing": [{"antennas": [0, 2]}, {"global_rescale": {}}]},
        {"extractors": [{"name": "fake_ext", "params": {"scale": 2}}]},
        {"dataset": "other"},
    ],
)
def test_key_part_changes(changes):
    if changes.get("dataset") == "other":
        base = registry.get("dataset", "tor_vergata")
        registry.register("dataset", "other")(type("Other", (base,), {}))
    assert key(changes) != key()


def test_key_changes_with_split_hash_and_fitted_state():
    assert key(split_hash="t") != key()
    assert key(fitted=[{"ids": [0, 1]}, {"mean_amplitude": 1.6}]) != key()


def test_key_changes_with_versions(monkeypatch):
    base = key()
    monkeypatch.setattr(registry.get("extractor", "fake_ext"), "VERSION", 2)
    assert key() != base
    monkeypatch.undo()
    monkeypatch.setattr(registry.get("dataset", "tor_vergata"), "VERSION", 99)
    assert key() != base
    monkeypatch.undo()
    monkeypatch.setattr(registry.get("preprocess", "antennas"), "VERSION", 99)
    assert key() != base


def test_key_changes_with_vendor_and_weights(monkeypatch):
    base = key()
    cls = registry.get("extractor", "fake_ext")
    monkeypatch.setattr(cls, "VENDOR_HASH", "abc", raising=False)
    with_vendor = key()
    assert with_vendor != base
    monkeypatch.setattr(cls, "WEIGHTS_REVISION", "r1", raising=False)
    assert key() != with_vendor


def test_window_geometry_only_keys_per_window_extractors():
    rec = {"extractors": [{"name": "fake_rec"}]}
    assert key({**rec, "window_len": 41}) == key(rec)
    assert key({**rec, "stride": 3}) == key(rec)
