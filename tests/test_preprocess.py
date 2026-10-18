import sys
from dataclasses import replace

import numpy as np
import pytest

from csi_bench.adapters.preprocess.amplitude_only import AmplitudeOnly
from csi_bench.adapters.preprocess.antennas import Antennas
from csi_bench.adapters.preprocess.global_rescale import GlobalRescale
from csi_bench.adapters.preprocess.per_packet_power import PerPacketPower
from csi_bench.adapters.preprocess.phase_sanitize import PhaseSanitize
from csi_bench.adapters.preprocess.recording_mean_norm import RecordingMeanNorm
from csi_bench.config.experiment import _expand_preprocessing
from csi_bench.config.presets import PREPROCESS_PRESETS
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
        PerPacketPower(),
        PhaseSanitize(),
        AmplitudeOnly(),
    ]


@pytest.mark.parametrize("index", range(len(_steps())))
def test_contract(index):
    check_preprocess_step(_steps()[index], _recs(), BLOCKS)


def test_registered_steps_all_covered():
    registry.reset()
    for m in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        del sys.modules[m]
    discover()
    assert registry.names("preprocess") == [
        "amplitude_only",
        "antennas",
        "global_rescale",
        "per_packet_power",
        "phase_sanitize",
        "recording_mean_norm",
    ]


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


# --- LWM preprocessing variants -------------------------------------------------------

N_SUB = 234


def _rec(csi: np.ndarray) -> Recording:
    csi = csi.astype(np.complex64)
    n_ant = csi.shape[0]
    return Recording(csi, tuple(range(n_ant)), "a", "r1", 0, RecordingMeta(csi.shape[1], "x"))


def _smooth_csi(rng: np.random.Generator, n_ant: int = 4, n_pkt: int = 20) -> np.ndarray:
    """CSI with varying amplitude and a smooth, non-linear phase (unwraps unambiguously)."""
    k = np.arange(N_SUB)
    amp = 1 + 0.5 * rng.random((n_ant, n_pkt, N_SUB))
    wiggle = 0.4 * np.sin(k / 15 + rng.uniform(0, 6, size=(n_ant, n_pkt, 1)))
    offsets = rng.uniform(-np.pi, np.pi, size=(n_ant, 1, 1))
    return amp * np.exp(1j * (wiggle + offsets))


def _slope(rng: np.random.Generator, n_pkt: int) -> np.ndarray:
    k = np.arange(N_SUB)
    a = rng.uniform(-0.6, 0.0, size=(n_pkt, 1))
    b = rng.uniform(-np.pi, np.pi, size=(n_pkt, 1))
    return np.exp(1j * (a * k + b))[None]


def test_per_packet_power_unit_power():
    rng = np.random.default_rng(0)
    csi = 1e12 * (rng.normal(size=(4, 50, 30)) + 1j * rng.normal(size=(4, 50, 30)))
    csi[1] *= 3  # antenna power ratio must survive
    out = PerPacketPower().transform(_rec(csi), {}).csi
    assert out.dtype == np.complex64
    power = (np.abs(out.astype(np.complex128)) ** 2).mean(axis=(0, 2))
    np.testing.assert_allclose(power, 1.0, rtol=1e-5)
    ratio = np.abs(out[1]) / np.abs(out[0])
    np.testing.assert_allclose(ratio, np.abs(csi[1]) / np.abs(csi[0]), rtol=1e-4)


def test_per_packet_power_zero_packet():
    rng = np.random.default_rng(1)
    csi = rng.normal(size=(4, 10, 30)) + 1j * rng.normal(size=(4, 10, 30))
    csi[:, 3] = 0
    out = PerPacketPower().transform(_rec(csi), {}).csi
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[:, 3], 0)
    power = (np.abs(np.delete(out, 3, axis=1)) ** 2).mean(axis=(0, 2))
    np.testing.assert_allclose(power, 1.0, rtol=1e-5)


def test_phase_sanitize_removes_slope():
    rng = np.random.default_rng(2)
    x = _smooth_csi(rng)
    sloped = x * _slope(rng, x.shape[1])
    step = PhaseSanitize()
    clean = step.transform(_rec(x), {}).csi
    out = step.transform(_rec(sloped), {}).csi
    assert out.dtype == np.complex64
    np.testing.assert_allclose(out, clean, atol=1e-4)
    np.testing.assert_allclose(np.abs(out), np.abs(x), rtol=1e-5)


def test_phase_sanitize_keeps_relative_antenna_phase():
    rng = np.random.default_rng(3)
    c = np.array([0.0, 1.0, -2.0, 2.9])
    amp = 1 + 0.5 * rng.random((4, 20, N_SUB))
    csi = amp * np.exp(1j * c)[:, None, None] * _slope(rng, 20)
    out = PhaseSanitize().transform(_rec(csi), {}).csi
    for r in range(4):
        rel = np.angle(out[r] * np.conj(out[0]))
        expected = np.angle(np.exp(1j * (c[r] - c[0])))
        diff = np.angle(np.exp(1j * (rel - expected)))
        np.testing.assert_allclose(diff, 0, atol=1e-4)


def test_phase_sanitize_pure_linear_phase_goes_to_zero():
    k = np.arange(N_SUB)
    csi = 2.0 * np.exp(1j * (-0.35 * k + 1.2)) * np.ones((4, 5, 1))
    out = PhaseSanitize().transform(_rec(csi), {}).csi
    np.testing.assert_allclose(np.angle(out), 0, atol=1e-4)
    np.testing.assert_allclose(np.abs(out), 2.0, rtol=1e-5)


def test_amplitude_only():
    rec = _recs()[0]
    out = AmplitudeOnly().transform(rec, {}).csi
    assert out.dtype == np.complex64
    np.testing.assert_array_equal(out.real, np.abs(rec.csi))
    np.testing.assert_array_equal(out.imag, 0)


LWM_PRESETS = {
    "lwm_rescale": ["global_rescale"],
    "lwm_power": ["per_packet_power"],
    "lwm_rescale_phase": ["global_rescale", "phase_sanitize"],
    "lwm_power_phase": ["per_packet_power", "phase_sanitize"],
    "lwm_amplitude": ["global_rescale", "amplitude_only"],
}


@pytest.mark.parametrize("preset", sorted(LWM_PRESETS))
def test_lwm_preset_composes_with_antennas(preset):
    steps = _expand_preprocessing(["ant2", preset])
    assert [(s["name"], s["params"]) for s in steps] == [
        ("antennas", {"ids": [0, 1]}),
        *[(name, {}) for name in LWM_PRESETS[preset]],
    ]


def test_lwm_presets_use_registered_steps():
    discover()
    names = set(registry.names("preprocess"))
    lwm = {k: v for k, v in PREPROCESS_PRESETS.items() if k.startswith("lwm_")}
    assert set(lwm) == set(LWM_PRESETS)
    for steps in lwm.values():
        assert {s["name"] for s in steps} <= names
