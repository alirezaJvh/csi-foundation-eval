"""Per-recording packet cache and its key (Story 5.3)."""

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from csi_bench.adapters.storage.packet_cache import NpzPacketCache
from csi_bench.config.experiment import config_from_dict, packet_cache_key
from csi_bench.core import registry
from csi_bench.core.contracts import PacketFeatures
from csi_bench.runner.discover import discover

BASE = {
    "dataset": "tor_vergata",
    "selection": {"rooms": ["r1"]},
    "preprocessing": ["ant2", "lwm_rescale"],
    "window_len": 10,
    "extractors": [{"name": "lwm"}],
    "pooler": "mean",
    "head": {"name": "logreg"},
    "role": "comparison",
}
FITTED = [{"ids": [0, 1]}, {"mean_amplitude": 1.5}]


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    discover()
    yield
    registry.reset()


def features() -> PacketFeatures:
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    return PacketFeatures(
        values=values,
        feature_names=("lwm.cls.0", "lwm.cls.1", "lwm.cls.2"),
        input_stats={"median_abs": 0.9, "p95_abs": 2.5},
    )


def key(changes=None, fitted=None, recording="tor_vergata/r1_empty", sha="abc") -> str:
    config = config_from_dict({**BASE, **(changes or {})})
    return packet_cache_key(config, config.extractors[0], fitted or FITTED, recording, sha)


def test_roundtrip(tmp_path):
    cache = NpzPacketCache(tmp_path)
    assert cache.get("k") is None
    cache.put("k", features())
    got = cache.get("k")
    assert got is not None
    np.testing.assert_array_equal(got.values, features().values)
    assert got.feature_names == features().feature_names
    assert got.input_stats == features().input_stats
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_entry_is_a_miss(tmp_path, caplog):
    cache = NpzPacketCache(tmp_path)
    cache.put("k", features())
    path = tmp_path / "k.npz"
    path.write_bytes(path.read_bytes()[:30])
    with caplog.at_level(logging.WARNING):
        assert cache.get("k") is None
    assert "unreadable" in caplog.text


def test_failed_write_leaves_nothing(tmp_path, monkeypatch):
    cache = NpzPacketCache(tmp_path)

    def boom(*args):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        cache.put("k", features())
    assert list(Path(tmp_path).iterdir()) == []


def test_key_ignores_window_pooler_rooms_head_and_seeds():
    assert key() == key(
        {
            "window_len": 40,
            "stride": 3,
            "pooler": "mean_std",
            "selection": {"rooms": ["r1", "r2"]},
            "head": {"name": "svm"},
            "seeds": [1, 2],
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"extractors": [{"name": "lwm", "params": {"embedding": "channel_mean"}}]},
        {"preprocessing": ["ant4", "lwm_rescale"]},
        {"preprocessing": ["ant2", "lwm_rescale_phase"]},
    ],
    ids=["embedding", "antennas", "preset"],
)
def test_key_changes_with_inputs(changes):
    assert key(changes) != key()


def test_key_changes_with_recording_and_fitted_state():
    assert key(recording="tor_vergata/r1_walking_1") != key()
    assert key(sha="def") != key()
    assert key(fitted=[{"ids": [0, 1]}, {"mean_amplitude": 2.0}]) != key()


def test_key_ignores_batch_size():
    batch = {"extractors": [{"name": "lwm", "params": {"batch_size": 7}}]}
    assert key(batch) == key()


def test_key_changes_with_vendor_and_weights(monkeypatch):
    lwm: Any = registry.get("extractor", "lwm")
    before = key()
    vendor = lwm.VENDOR_HASH
    monkeypatch.setattr(lwm, "VENDOR_HASH", "other")
    assert key() != before
    monkeypatch.setattr(lwm, "VENDOR_HASH", vendor)
    monkeypatch.setattr(lwm, "WEIGHTS_REVISION", "other")
    assert key() != before
