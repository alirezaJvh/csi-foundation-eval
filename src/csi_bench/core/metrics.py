"""The Evaluator: one pure function that scores val and test predictions (AD-11)."""

from collections.abc import Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from csi_bench.core.contracts import Metrics, Predictions


def evaluate(predictions: Predictions, class_names: Sequence[str]) -> Metrics:
    """Accuracy, macro-F1, per-class precision/recall/F1 and the confusion matrix."""
    labels = list(range(len(class_names)))
    y_true, y_pred = predictions.y_true, predictions.y_pred
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return Metrics(
        split=predictions.split,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        per_class=per_class,
        confusion_matrix=tuple(tuple(int(v) for v in row) for row in matrix),
        class_names=tuple(class_names),
    )
