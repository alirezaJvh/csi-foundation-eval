import pytest

from csi_bench.core import results_schema as rs
from csi_bench.core.contracts import Metrics

CLASSES = ("empty", "sitting")


def _metrics(split: str, accuracy: float, macro_f1: float) -> Metrics:
    return Metrics(
        split=split,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class={c: {"precision": 1.0, "recall": 1.0, "f1": 1.0} for c in CLASSES},
        confusion_matrix=((1, 0), (0, 1)),
        class_names=CLASSES,
    )


def test_mean_std_over_two_seeds():
    doc = rs.metrics_document(
        {
            0: {"val": _metrics("val", 0.5, 0.4), "test": _metrics("test", 0.6, 0.5)},
            1: {"val": _metrics("val", 0.7, 0.6), "test": _metrics("test", 0.6, 0.5)},
        }
    )
    assert doc["class_names"] == list(CLASSES)
    assert set(doc["seeds"]) == {"0", "1"}
    assert doc["mean_std"]["val"]["accuracy"] == pytest.approx([0.6, 0.1])
    assert doc["mean_std"]["val"]["macro_f1"] == pytest.approx([0.5, 0.1])
    assert doc["mean_std"]["test"]["accuracy"] == pytest.approx([0.6, 0.0])


def test_mean_std_single_seed_has_zero_std():
    doc = rs.metrics_document({0: {"val": _metrics("val", 0.8, 0.7)}})
    assert doc["mean_std"]["val"]["accuracy"] == pytest.approx([0.8, 0.0])


def test_metrics_round_trip():
    from csi_bench.core.contracts import Metrics

    m = Metrics(
        split="val",
        accuracy=0.5,
        macro_f1=0.4,
        per_class={"a": {"precision": 1.0, "recall": 0.5, "f1": 0.6}},
        confusion_matrix=((1, 2), (3, 4)),
        class_names=("a", "b"),
    )
    assert rs.metrics_from_dict(rs.metrics_as_dict(m)) == m
