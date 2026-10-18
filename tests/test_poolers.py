"""Weight-free poolers over per-packet features (Story 5.3)."""

import sys
from typing import Any

import numpy as np
import pytest

from csi_bench.core import registry
from csi_bench.core.contracts import PacketFeatures
from csi_bench.runner.discover import discover
from tests.contracts.pooler import check_pooler


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    discover()
    yield
    registry.reset()


def _packets(n=120, d=5, offset=0.0, seed=0) -> PacketFeatures:
    rng = np.random.default_rng(seed)
    values = (offset + rng.normal(0, 1, (n, d))).astype(np.float32)
    return PacketFeatures(
        values=values,
        feature_names=tuple(f"lwm.cls.{i}" for i in range(d)),
        input_stats={"median_abs": 1.0},
    )


def _pooler(name):
    cls: Any = registry.get("pooler", name)
    return cls(cls.Params())


def test_registered_poolers():
    assert registry.names("pooler") == ["mean", "mean_std"]


@pytest.mark.parametrize("name", ["mean", "mean_std"])
@pytest.mark.parametrize("window_len", [2, 10, 50])
def test_contract(name, window_len):
    check_pooler(_pooler(name), _packets(), window_len)


def test_mean_window_len_1_is_the_packet_itself():
    packets = _packets(n=20)
    pooled = _pooler("mean").pool(packets, list(range(20)), 1)
    np.testing.assert_array_equal(pooled.values, packets.values)
    assert pooled.feature_names[0] == "lwm.cls.0.mean"


def test_mean_std_names_and_min_window_len():
    pooled = _pooler("mean_std").pool(_packets(d=2), [0], 4)
    assert pooled.feature_names == (
        "lwm.cls.0.mean",
        "lwm.cls.1.mean",
        "lwm.cls.0.std",
        "lwm.cls.1.std",
    )
    mean: Any = registry.get("pooler", "mean")
    mean_std: Any = registry.get("pooler", "mean_std")
    assert (mean.min_window_len, mean_std.min_window_len) == (1, 2)


def test_mean_std_is_stable_for_large_offsets():
    # long recordings with a large constant offset must not cancel E[x²] - E[x]²
    check_pooler(_pooler("mean_std"), _packets(n=6000, offset=50.0), 40)
