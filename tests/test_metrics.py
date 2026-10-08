import numpy as np

from csi_bench.core.contracts import Predictions
from csi_bench.core.metrics import evaluate

CLASSES = ("empty", "sitting", "standing", "walking")


def _pred(y_true, y_pred, split="val"):
    return Predictions(
        window_ids=tuple(f"r/{i}" for i in range(len(y_true))),
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        split=split,
    )


def test_perfect_predictions():
    y = [0, 1, 2, 3, 0, 1, 2, 3]
    metrics = evaluate(_pred(y, y, "test"), CLASSES)
    assert metrics.split == "test"
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert set(metrics.per_class) == set(CLASSES)
    assert metrics.per_class["standing"]["precision"] == 1.0
    assert metrics.confusion_matrix == ((2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2))


def test_confusion_and_per_class():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    metrics = evaluate(_pred(y_true, y_pred), CLASSES)
    assert metrics.confusion_matrix[0] == (1, 1, 0, 0)
    assert metrics.confusion_matrix[1] == (0, 2, 0, 0)
    assert metrics.per_class["empty"]["precision"] == 1.0
    assert metrics.per_class["empty"]["recall"] == 0.5
    assert metrics.per_class["sitting"]["support"] == 2
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
