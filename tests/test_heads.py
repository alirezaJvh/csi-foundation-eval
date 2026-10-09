import sys
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from csi_bench.core import head_search, registry
from csi_bench.core.contracts import FeatureMatrix
from csi_bench.runner.discover import discover
from tests.contracts.head import check_head, check_scaler_inside, check_seed_determinism

NAMES = ("f.a", "f.b", "f.c")
HEADS = ["knn", "logreg", "mlp", "random_forest", "svm"]
BAD_PARAMS = {
    "svm": {"C_grid": ()},
    "knn": {"k_grid": ()},
    "random_forest": {"n_estimators_grid": ()},
    "mlp": {"alpha_grid": ()},
    "logreg": {"C_grid": ()},
}
SMALL = {
    "svm": {"C_grid": [1.0, 10.0]},
    "knn": {"k_grid": [1, 3]},
    "random_forest": {"n_estimators_grid": [10], "max_depth_grid": [None, 3]},
    "mlp": {"alpha_grid": [1e-4, 1e-2], "max_iter": 200},
    "logreg": {"C_grid": [0.1, 1.0]},
}


def _part(n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    vectors, labels = [], []
    for cls in range(4):
        vectors.append(rng.normal(loc=cls, scale=0.4, size=(n_per_class, 3)))
        labels.extend([cls] * n_per_class)
    values = np.vstack(vectors).astype(np.float32)
    return FeatureMatrix(values=values, feature_names=NAMES), np.array(labels)


@pytest.fixture(autouse=True)
def discovered(monkeypatch):
    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    discover()
    yield
    registry.reset()


def _make(name):
    cls: Any = registry.get("head", name)
    return lambda seed: cls(cls.Params(**SMALL[name]), seed)


def test_all_heads_registered():
    assert registry.names("head") == HEADS


@pytest.mark.parametrize("name", HEADS)
def test_contract_accuracy_and_determinism(name):
    train, y_train = _part(60, 0)
    val, y_val = _part(30, 1)
    test, y_test = _part(30, 2)
    make = _make(name)
    head = make(0)
    check_head(head, train, y_train, val, y_val, test)
    assert float((head.predict(test) == y_test).mean()) > 0.9
    check_seed_determinism(make, train, y_train, val, y_val, test)
    check_scaler_inside(make, _part)


@pytest.mark.parametrize("name", HEADS)
def test_params_reject_unknown_and_empty_grid(name):
    cls: Any = registry.get("head", name)
    with pytest.raises(ValidationError):
        cls.Params(nope=1)
    with pytest.raises(ValidationError):
        cls.Params(**BAD_PARAMS[name])


def test_svm_kernel_and_forest_depth_validation():
    svm: Any = registry.get("head", "svm")
    forest_cls: Any = registry.get("head", "random_forest")
    with pytest.raises(ValidationError):
        svm.Params(kernel="poly")
    forest = forest_cls.Params(max_depth_grid=[None, 5])
    assert forest.max_depth_grid == (None, 5)


@pytest.mark.parametrize("name", HEADS)
def test_test_labels_never_reach_fit(name):
    train, y_train = _part(40, 0)
    val, y_val = _part(20, 1)

    class Recorder:
        def __init__(self, head):
            self._head = head
            self.seen: list[np.ndarray] = []

        def fit(self, x_train, seen_y_train, x_val, seen_y_val):
            self.seen.extend([np.asarray(seen_y_train), np.asarray(seen_y_val)])
            self._head.fit(x_train, seen_y_train, x_val, seen_y_val)

        def predict(self, features: FeatureMatrix) -> np.ndarray:
            return self._head.predict(features)

    recorder = Recorder(_make(name)(0))
    recorder.fit(train, y_train, val, y_val)

    # fit is handed exactly the train and val labels and nothing else — no test labels.
    assert len(recorder.seen) == 2
    np.testing.assert_array_equal(recorder.seen[0], y_train)
    np.testing.assert_array_equal(recorder.seen[1], y_val)


def test_grid_selected_on_val_only_and_refit_on_union():
    train, y_train = _part(40, 0)
    val, y_val = _part(20, 1)

    class Fixed:
        def __init__(self, good: bool):
            self.good = good

        def fit(self, x, y):
            self.rows = len(x)
            self.classes = np.unique(y)
            return self

        def predict(self, x):
            return np.zeros(len(x), dtype=int)

    fits: list[Fixed] = []

    def candidate(good):
        def make():
            model = Fixed(good)
            fits.append(model)
            return model

        return make

    # both predict zeros -> tie -> first wins
    scaler, model, index = head_search.tune_and_refit(
        [candidate(1), candidate(2)], train, y_train, val, y_val
    )
    assert index == 0
    assert model.rows == len(train.values) + len(val.values)
    assert scaler.n_samples_seen_ == len(train.values) + len(val.values)

    # a candidate that matches val labels wins even when listed second
    class Oracle(Fixed):
        def predict(self, x):
            return np.repeat(y_val, 1)[: len(x)] if len(x) == len(y_val) else np.zeros(len(x), int)

    _, _, index = head_search.tune_and_refit(
        [candidate(1), lambda: Oracle(True)], train, y_train, val, y_val
    )
    assert index == 1
