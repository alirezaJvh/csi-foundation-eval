import sys
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from csi_bench.core import registry
from csi_bench.core.contracts import FeatureMatrix
from csi_bench.core.errors import ConfigError
from csi_bench.runner.discover import discover
from tests.contracts.selector import check_selector

NAMES = ("fam_a.x", "fam_a.y", "fam_b.signal", "fam_b.noise")


@pytest.fixture(autouse=True)
def discovered(monkeypatch):
    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    discover()
    yield
    registry.reset()


def _part(n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1, 2], n_per_class)
    values = rng.normal(size=(len(labels), 4))
    values[:, 2] += labels * 3  # fam_b.signal separates the classes
    values[:, 0] += labels * 0.5
    return FeatureMatrix(values=values.astype(np.float32), feature_names=NAMES), labels


def _make(name: str, **params):
    cls: Any = registry.get("selector", name)
    return cls(cls.Params(**params))


def test_registered():
    assert registry.names("selector") == ["manual", "mrmr"]


@pytest.mark.parametrize(
    ("name", "params"),
    [("manual", {"columns": ["fam_a", "fam_b.noise"]}), ("mrmr", {"k": [1, 2, 3]})],
)
def test_contract(name, params):
    (train, y_train), (val, y_val), (test, _) = _part(40, 0), _part(20, 1), _part(20, 2)
    check_selector(_make(name, **params), train, y_train, val, y_val, test)


def test_manual_names_and_families_in_matrix_order():
    (train, y_train), (val, y_val) = _part(10, 0), _part(5, 1)
    selector = _make("manual", columns=["fam_b.noise", "fam_a"])
    selector.fit(train, y_train, val, y_val)
    assert selector.selected_names == ("fam_a.x", "fam_a.y", "fam_b.noise")
    assert selector.transform(train).feature_names == selector.selected_names


def test_manual_family_prefix_is_not_a_substring_match():
    with pytest.raises(ConfigError, match="fam"):
        _make("manual", columns=["fam"]).validate(NAMES)


def test_manual_unknown_entry_names_entry_and_columns():
    with pytest.raises(ConfigError) as info:
        _make("manual", columns=["fam_a.x", "nope"]).validate(NAMES)
    assert "nope" in str(info.value)
    assert "fam_b.signal" in str(info.value)


def test_manual_bad_params():
    for bad in ({"columns": []}, {}, {"columns": ["a"], "extra": 1}):
        with pytest.raises(ValidationError):
            _make("manual", **bad)


def test_mrmr_bad_params():
    for bad in ({"k": []}, {"k": [0]}, {"k": [2, 2]}, {}):
        with pytest.raises(ValidationError):
            _make("mrmr", **bad)


def test_mrmr_k_too_large():
    with pytest.raises(ConfigError, match="exceeds"):
        _make("mrmr", k=[2, 5]).validate(NAMES)


def test_mrmr_candidates_are_prefixes_of_one_train_ranking():
    (train, y_train), (val, y_val) = _part(40, 0), _part(20, 1)
    selector = _make("mrmr", k=[3, 1, 2])
    selector.fit(train, y_train, val, y_val)
    three, one, two = selector.candidates
    assert (len(three), len(one), len(two)) == (3, 1, 2)
    assert three[:1] == one and three[:2] == two
    assert one == ("fam_b.signal",)


def test_mrmr_ranking_ignores_val_rows():
    (train, y_train), (val, y_val) = _part(40, 0), _part(20, 1)
    first = _make("mrmr", k=[3])
    first.fit(train, y_train, val, y_val)
    second = _make("mrmr", k=[3])
    scrambled = FeatureMatrix(values=-val.values[::-1], feature_names=NAMES)
    second.fit(train, y_train, scrambled, y_val[::-1])
    assert first.candidates == second.candidates
