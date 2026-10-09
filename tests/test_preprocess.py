import sys
from dataclasses import replace

import numpy as np
import pytest

from csi_bench.adapters.preprocess.antennas import Antennas
from csi_bench.adapters.preprocess.global_rescale import GlobalRescale
from csi_bench.adapters.preprocess.recording_mean_norm import RecordingMeanNorm
from csi_bench.core import registry
from csi_bench.core.contracts import Recording, RecordingMeta
from csi_bench.core.errors import ConfigError, DataError
from csi_bench.core.preprocess import run_chain, train_view
from csi_bench.runner.discover import discover
from tests.contracts.preprocess_step import check_preprocess_step

BLOCKS = {
    "a": {"train": (0, 40), "val": (50, 70), "test": (80, 100)},
    "b": {"train": (0, 40), "val": (50, 70), "test": (80, 100)},
}


def _recs() -> list[Recording]:
    out = []
    for i, rid in enumerate(("a", "b")):
        rng = np.random.default_rng(i)
        csi = (1 + i + rng.normal(size=(4, 100, 3)) + 1j * rng.normal(size=(4, 100, 3))).astype(
            np.complex64
        )
        out.append(Recording(csi, (0, 1, 2, 3), rid, "r1", i, RecordingMeta(100, "x")))
    return out


def _steps():
    return [
        Antennas(Antennas.Params(ids=[2, 0])),
        GlobalRescale(),
        RecordingMeanNorm(),
    ]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_contract(index):
    check_preprocess_step(_steps()[index], _recs(), BLOCKS)


def test_registered_steps_all_covered():
    registry.reset()
    for m in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        del sys.modules[m]
    discover()
    assert registry.names("preprocess") == ["antennas", "global_rescale", "recording_mean_norm"]


def test_antennas_order_and_ids():
    out, states = run_chain([Antennas(Antennas.Params(ids=[2, 0]))], _recs(), BLOCKS)
    assert out[0].antenna_ids == (2, 0)
    np.testing.assert_array_equal(out[0].csi[0], _recs()[0].csi[2])
    assert states == [{"ids": [2, 0]}]


def test_antennas_missing_id():
    with pytest.raises(ConfigError):
        run_chain([Antennas(Antennas.Params(ids=[0, 7]))], _recs(), BLOCKS)


def test_global_rescale_uses_train_only():
    recs = _recs()
    out, states = run_chain([GlobalRescale()], recs, BLOCKS)
    train = np.concatenate([np.abs(r.csi[:, :40]).ravel() for r in recs])
    expected = float(train.astype(np.float64).mean())
    assert states[0]["mean_amplitude"] == pytest.approx(expected)
    assert out[0].csi.dtype == np.complex64
    np.testing.assert_allclose(out[0].csi, recs[0].csi / expected, rtol=1e-5)


def test_chain_order_matters():
    recs = _recs()
    _, a = run_chain([Antennas(Antennas.Params(ids=[0, 1])), GlobalRescale()], recs, BLOCKS)
    _, b = run_chain([GlobalRescale(), Antennas(Antennas.Params(ids=[0, 1]))], recs, BLOCKS)
    assert a[1] != b[0]


def test_recording_mean_norm_per_antenna():
    recs = _recs()
    out, states = run_chain([RecordingMeanNorm()], recs, BLOCKS)
    train = out[0].csi[:, :40]
    np.testing.assert_allclose(np.abs(train).mean(axis=(1, 2)), 1.0, rtol=1e-5)
    assert set(states[0]["means"]) == {"a", "b"}


def test_whole_recording_scope_sees_everything():
    step = RecordingMeanNorm(RecordingMeanNorm.Params(fit_scope="whole_recording"))
    out, _ = run_chain([step], _recs(), BLOCKS)
    np.testing.assert_allclose(np.abs(out[0].csi).mean(axis=(1, 2)), 1.0, rtol=1e-5)


def test_non_positive_mean_is_data_error():
    rec = replace(_recs()[0], csi=np.zeros((4, 100, 3), dtype=np.complex64))
    with pytest.raises(DataError):
        GlobalRescale().fit(train_view([rec], BLOCKS))
    with pytest.raises(DataError):
        RecordingMeanNorm().fit(train_view([rec], BLOCKS))
