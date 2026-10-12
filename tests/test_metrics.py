import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from csi_bench.core.contracts import Predictions
from csi_bench.core.errors import DataError
from csi_bench.core.metrics import bootstrap_intervals, evaluate

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


# --- bootstrap_intervals -------------------------------------------------------------------


def _recordings(labels, windows=5):
    """`(recording_id, y_true)` with `windows` windows per recording; ids `rec<i>`."""
    ids = np.repeat([f"rec{i}" for i in range(len(labels))], windows)
    return ids, np.repeat(np.asarray(labels, dtype=np.int64), windows)


def _spread(seed=0):
    """4 classes x 3 recordings, with errors spread over some recordings."""
    ids, y_true = _recordings([k for k in range(4) for _ in range(3)])
    rng = np.random.default_rng(seed)
    y_pred = y_true.copy()
    wrong = rng.random(len(y_pred)) < 0.25
    y_pred[wrong] = (y_pred[wrong] + rng.integers(1, 4, wrong.sum())) % 4
    return ids, y_true, y_pred


def test_bootstrap_all_correct_is_a_point():
    ids, y = _recordings([0, 0, 1, 1, 2, 2, 3, 3])
    result = bootstrap_intervals([(ids, y, y.copy())], 4, n_resamples=200)
    assert result["accuracy"] == [1.0, 1.0]
    assert result["macro_f1"] == [1.0, 1.0]
    assert result["single_recording_classes"] == []
    assert result["unit"] == "recording" and result["n_resamples"] == 200
    assert result["seed"] == 0 and result["level"] == 0.95


def test_bootstrap_spread_contains_point_and_has_width():
    ids, y_true, y_pred = _spread()
    result = bootstrap_intervals([(ids, y_true, y_pred)], 4, class_names=CLASSES)
    point = evaluate(_pred(y_true, y_pred, "test"), CLASSES)
    for score, value in (("accuracy", point.accuracy), ("macro_f1", point.macro_f1)):
        lo, hi = result[score]
        assert lo <= value <= hi
        assert lo < hi


def test_bootstrap_flags_single_recording_class():
    ids, y_true = _recordings([0, 1, 1, 2, 2, 3, 3])
    y_pred = y_true.copy()
    y_pred[:2] = 1
    result = bootstrap_intervals([(ids, y_true, y_pred)], 4, class_names=CLASSES)
    assert result["single_recording_classes"] == ["empty"]
    assert bootstrap_intervals([(ids, y_true, y_pred)], 4)["single_recording_classes"] == ["0"]


def test_bootstrap_two_seeds_cover_the_seed_mean():
    ids, y_true, pred_a = _spread(1)
    _, _, pred_b = _spread(2)
    result = bootstrap_intervals([(ids, y_true, pred_a), (ids, y_true, pred_b)], 4)
    for score, fn in (
        ("accuracy", accuracy_score),
        ("macro_f1", lambda t, p: f1_score(t, p, average="macro", zero_division=0)),
    ):
        mean = (fn(y_true, pred_a) + fn(y_true, pred_b)) / 2
        lo, hi = result[score]
        assert lo <= mean <= hi


@pytest.mark.parametrize("pred_seeds", [(3,), (3, 4)])
def test_bootstrap_matches_window_rescoring_with_the_same_draw(pred_seeds):
    """Confusion-matrix sums equal sklearn on the resampled windows, absent class included.

    With several seeds, one draw serves every seed and the statistic is the per-resample seed
    mean, not one seed's score or an average of per-seed bounds.
    """
    ids, y_true, _ = _spread(3)
    preds = [_spread(s)[2] for s in pred_seeds]
    n_classes, n_resamples, seed = 5, 50, 7  # class 4 never occurs: macro over all 5 classes
    result = bootstrap_intervals([(ids, y_true, p) for p in preds], n_classes, n_resamples, seed)
    recs = np.unique(ids)
    labels = np.array([y_true[ids == r][0] for r in recs])
    rng = np.random.default_rng(seed)
    draws = np.zeros((n_resamples, 0), dtype=int)
    for k in range(n_classes):
        members = np.flatnonzero(labels == k)
        if members.size:
            pick = members[rng.integers(0, members.size, size=(n_resamples, members.size))]
            draws = np.concatenate([draws, pick], axis=1)
    accs, f1s = [], []
    for b in range(n_resamples):
        rows = np.concatenate([np.flatnonzero(ids == recs[r]) for r in draws[b]])
        accs.append(np.mean([accuracy_score(y_true[rows], p[rows]) for p in preds]))
        f1s.append(
            np.mean(
                [
                    f1_score(
                        y_true[rows],
                        p[rows],
                        labels=list(range(n_classes)),
                        average="macro",
                        zero_division=0,
                    )
                    for p in preds
                ]
            )
        )
    assert result["accuracy"] == pytest.approx(list(np.percentile(accs, [2.5, 97.5])))
    assert result["macro_f1"] == pytest.approx(list(np.percentile(f1s, [2.5, 97.5])))


def test_bootstrap_is_deterministic():
    ids, y_true, y_pred = _spread()
    first = bootstrap_intervals([(ids, y_true, y_pred)], 4, n_resamples=300, seed=5)
    again = bootstrap_intervals([(ids, y_true, y_pred)], 4, n_resamples=300, seed=5)
    assert first == again
    other = bootstrap_intervals([(ids, y_true, y_pred)], 4, n_resamples=300, seed=6)
    assert other != first


def test_bootstrap_rejects_seeds_with_different_recordings():
    ids, y_true, y_pred = _spread()
    keep = ids != "rec0"
    with pytest.raises(DataError, match="disagree"):
        bootstrap_intervals([(ids, y_true, y_pred), (ids[keep], y_true[keep], y_pred[keep])], 4)


def test_bootstrap_rejects_mixed_label_recording():
    ids, y_true, y_pred = _spread()
    y_true = y_true.copy()
    y_true[0] = (y_true[0] + 1) % 4
    with pytest.raises(DataError, match="mixed labels: rec0"):
        bootstrap_intervals([(ids, y_true, y_pred)], 4)


def test_bootstrap_rejects_empty_and_out_of_range():
    ids, y_true, y_pred = _spread()
    with pytest.raises(DataError):
        bootstrap_intervals([], 4)
    with pytest.raises(DataError):
        bootstrap_intervals([(ids[:0], y_true[:0], y_pred[:0])], 4)
    with pytest.raises(DataError, match="outside"):
        bootstrap_intervals([(ids, y_true, y_pred)], 3)
