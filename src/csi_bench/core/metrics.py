"""The Evaluator: one pure function that scores val and test predictions (AD-11)."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from csi_bench.core.contracts import Metrics, Predictions
from csi_bench.core.errors import DataError


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


def _recording_counts(
    recording_id: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(recording ids, their labels, per-recording confusion matrices (R, K, K))` of one seed."""
    recording_id, y_true, y_pred = (np.asarray(a) for a in (recording_id, y_true, y_pred))
    if not (recording_id.shape == y_true.shape == y_pred.shape) or recording_id.ndim != 1:
        raise DataError("bootstrap: recording_id, y_true and y_pred must be 1-D of equal length")
    if recording_id.size == 0:
        raise DataError("bootstrap: no test predictions")
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.issubdtype(values.dtype, np.integer):
            raise DataError(f"bootstrap: {name} must hold integer class indices")
        if values.min() < 0 or values.max() >= n_classes:
            raise DataError(f"bootstrap: {name} outside [0, {n_classes})")
    ids, rec = np.unique(recording_id, return_inverse=True)
    n_rec = len(ids)
    labels = np.full(n_rec, -1, dtype=np.int64)
    labels[rec] = y_true
    if np.any(labels[rec] != y_true):
        mixed = sorted({str(ids[r]) for r in rec[labels[rec] != y_true]})
        raise DataError(f"bootstrap: recordings with mixed labels: {', '.join(mixed)}")
    flat = (rec * n_classes + y_true) * n_classes + y_pred
    counts = np.bincount(flat, minlength=n_rec * n_classes * n_classes)
    return ids, labels, counts.reshape(n_rec, n_classes, n_classes)


def bootstrap_intervals(
    predictions: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_classes: int,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
    class_names: Sequence[str] | None = None,
) -> dict:
    """Percentile intervals of the seed-mean test accuracy and macro-F1 over recordings.

    `predictions` holds one `(recording_id, y_true, y_pred)` triple per seed, all over the same
    recordings. Each resample draws, per class, as many recordings as that class has, with
    replacement; the same draw serves every seed. Scores come from summed per-recording
    confusion matrices. `class_names` names the entries of `single_recording_classes`
    (default: the class indices as strings).
    """
    if not predictions:
        raise DataError("bootstrap: no seeds")
    if n_classes < 1 or n_resamples < 1 or not 0 < level < 1:
        raise DataError("bootstrap: need n_classes >= 1, n_resamples >= 1 and 0 < level < 1")
    names = list(class_names) if class_names is not None else [str(k) for k in range(n_classes)]
    if len(names) != n_classes:
        raise DataError(f"bootstrap: {len(names)} class names for {n_classes} classes")
    per_seed = [_recording_counts(*triple, n_classes) for triple in predictions]
    ids, labels, _ = per_seed[0]
    for other_ids, other_labels, _ in per_seed[1:]:
        if len(other_ids) != len(ids) or np.any(other_ids != ids):
            raise DataError("bootstrap: seeds disagree on the test recordings")
        if np.any(other_labels != labels):
            raise DataError("bootstrap: seeds disagree on the recording labels")
    counts = np.stack([c for _, _, c in per_seed]).astype(np.float64)  # (S, R, K, K)
    n_rec = len(ids)

    rng = np.random.default_rng(seed)
    weights = np.zeros((n_resamples, n_rec), dtype=np.float64)
    rows = np.arange(n_resamples)[:, None]
    single: list[str] = []
    for k in range(n_classes):
        members = np.flatnonzero(labels == k)
        if members.size == 0:
            continue
        if members.size == 1:
            single.append(names[k])
        draws = members[rng.integers(0, members.size, size=(n_resamples, members.size))]
        np.add.at(weights, (np.broadcast_to(rows, draws.shape), draws), 1.0)

    matrices = np.einsum("br,srkl->sbkl", weights, counts)  # (S, B, K, K)
    tp = np.diagonal(matrices, axis1=2, axis2=3)
    total = matrices.sum(axis=(2, 3))
    accuracy = tp.sum(axis=2) / total
    denom = matrices.sum(axis=3) + matrices.sum(axis=2)  # true + predicted per class
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    macro_f1 = f1.mean(axis=2)

    tail = 100 * (1 - level) / 2
    bounds = [tail, 100 - tail]

    def interval(stat: np.ndarray) -> list[float]:
        lo, hi = np.percentile(stat.mean(axis=0), bounds)
        return [float(lo), float(hi)]

    return {
        "accuracy": interval(accuracy),
        "macro_f1": interval(macro_f1),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "level": float(level),
        "unit": "recording",
        "single_recording_classes": single,
    }
