import numpy as np

from csi_bench.adapters.heads.logreg import LogReg
from csi_bench.core.contracts import FeatureMatrix
from tests.contracts.head import check_head

NAMES = ("f.a", "f.b", "f.c")


def _part(n_per_class: int, seed: int, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    vectors = []
    labels = []
    for cls in range(4):
        vectors.append(rng.normal(loc=cls, scale=0.4, size=(n_per_class, 3)))
        labels.extend([cls] * n_per_class)
    values = (np.vstack(vectors) * scale).astype(np.float32)
    return FeatureMatrix(values=values, feature_names=NAMES), np.array(labels)


def test_contract_and_accuracy():
    train, y_train = _part(60, 0)
    val, y_val = _part(30, 1)
    test, y_test = _part(30, 2)
    head = LogReg(seed=0)
    check_head(head, train, y_train, val, y_val, test)
    accuracy = float((head.predict(test) == y_test).mean())
    assert accuracy > 0.9


def test_seed_is_deterministic():
    train, y_train = _part(50, 0)
    val, y_val = _part(20, 1)
    test, _ = _part(20, 2)
    first = LogReg(seed=7)
    second = LogReg(seed=7)
    first.fit(train, y_train, val, y_val)
    second.fit(train, y_train, val, y_val)
    np.testing.assert_array_equal(first.predict(test), second.predict(test))


def test_scaling_happens_inside_the_head():
    train, y_train = _part(60, 0, scale=1e4)
    val, y_val = _part(30, 1, scale=1e4)
    test, y_test = _part(30, 2, scale=1e4)
    head = LogReg(seed=0)
    head.fit(train, y_train, val, y_val)
    assert float((head.predict(test) == y_test).mean()) > 0.9


def test_registered_via_discovery(monkeypatch):
    import sys

    from csi_bench.core import registry
    from csi_bench.runner.discover import discover

    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters.heads")]:
        monkeypatch.delitem(sys.modules, name)
    discover()
    assert registry.get("head", "logreg").__name__ == "LogReg"


def test_final_scaler_is_fitted_on_train_and_val():
    train, y_train = _part(60, 0)
    val, y_val = _part(30, 1, scale=3.0)
    head = LogReg(seed=0)
    head.fit(train, y_train, val, y_val)
    both = np.vstack([train.values, val.values])
    scaler = head._scaler
    assert scaler is not None and scaler.mean_ is not None
    assert np.allclose(scaler.mean_, both.mean(axis=0), rtol=1e-5)


def test_params_reject_unknown_and_empty_grid():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LogReg.Params(C_grd=(1.0,))  # ty: ignore[unknown-argument]
    with pytest.raises(ValidationError):
        LogReg.Params(C_grid=())
