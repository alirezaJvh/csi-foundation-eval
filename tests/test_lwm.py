"""Frozen LWM v1.1 per-packet extractor (Story 5.1)."""

import ast
import hashlib
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from csi_bench.adapters.extractors.lwm import adapter as lwm_adapter
from csi_bench.adapters.extractors.lwm._vendor.lwm_model import lwm
from csi_bench.adapters.extractors.lwm._vendor.patch_maker import patch_maker
from csi_bench.adapters.extractors.lwm.adapter import LWM, load_model, strip_module_prefix
from csi_bench.config.experiment import (
    compute_run_id,
    config_from_dict,
    resolved_body,
)
from csi_bench.core import registry
from csi_bench.core.errors import ConfigError, ExtractorError
from csi_bench.runner.discover import discover
from tests.contracts.extractor import check_packet_extractor

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "src/csi_bench/adapters/extractors/lwm/_vendor"
SHA = "1b68512dbe97de56841301206eacb780fcd76011"
LWM_MODEL_SHA256 = "472373a08c22c107a6135e5637fd0f92310789af5faa4e6d586e31f32de0aae5"
RECORDINGS = ROOT / "data/csi_proc_python/saved_csi"
BASE = {
    "dataset": "tor_vergata",
    "selection": {"rooms": ["r1"]},
    "window_len": 40,
    "extractors": [{"name": "amplitude_stats"}],
    "head": {"name": "logreg"},
    "role": "comparison",
}


@pytest.fixture(scope="module")
def model():
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)  # the weights cache is relative to the working directory
        return load_model()


@pytest.fixture
def make(model):
    """Extractor that reuses the module's loaded model (loading is covered separately)."""

    def build(**params) -> LWM:
        extractor = LWM(LWM.Params(**params))
        extractor._model = model
        return extractor

    return build


@pytest.fixture
def discovered(monkeypatch):
    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.chdir(ROOT)
    discover()
    yield
    registry.reset()


def _csi(n_antennas: int = 4, n_packets: int = 6, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    shape = (n_antennas, n_packets, 234)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return (csi * scale / np.sqrt(2)).astype(np.complex64)


def _checksum(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


# --- vendored code -------------------------------------------------------------------------


def test_vendor_folder_holds_only_the_pinned_excerpt():
    files = {p.name for p in VENDOR.iterdir() if p.name != "__pycache__"}
    assert files == {"__init__.py", "lwm_model.py", "patch_maker.py", "SOURCE.md"}
    assert set(lwm_adapter.VENDOR_FILES) == files
    assert (VENDOR / "__init__.py").read_bytes() == b""
    model_bytes = (VENDOR / "lwm_model.py").read_bytes()
    assert hashlib.sha256(model_bytes).hexdigest() == LWM_MODEL_SHA256
    for path in VENDOR.glob("*.py"):
        assert "deepmimo" not in path.read_text().lower(), path.name

    tree = ast.parse((VENDOR / "patch_maker.py").read_text())
    imports = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    defs = [n for n in tree.body if not isinstance(n, ast.Import | ast.ImportFrom)]
    assert [ast.unparse(n) for n in imports] == ["import numpy as np"]
    assert [type(n) for n in defs] == [ast.FunctionDef]
    assert isinstance(defs[0], ast.FunctionDef) and defs[0].name == "patch_maker"

    source = (VENDOR / "SOURCE.md").read_text()
    assert SHA in source and "https://huggingface.co/wi-lab/lwm-v1.1" in source
    for arg, value in [
        ("element_length", "32"),
        ("d_model", "128"),
        ("n_layers", "12"),
        ("max_len", "513"),
        ("n_heads", "8"),
        ("dropout", "0.1"),
    ]:
        assert f"`{arg}` | {value}" in source, arg


def test_vendor_hash_and_revision_are_exposed():
    assert LWM.WEIGHTS_REVISION == SHA
    assert LWM.VENDOR_HASH == lwm_adapter.vendor_hash()
    assert len(LWM.VENDOR_HASH) == 64


def test_vendor_hash_changes_with_file_bytes(tmp_path, monkeypatch):
    vendor = shutil.copytree(
        VENDOR, tmp_path / "_vendor", ignore=shutil.ignore_patterns("__pycache__")
    )
    monkeypatch.setattr(lwm_adapter, "VENDOR_DIR", vendor)
    assert lwm_adapter.vendor_hash() == LWM.VENDOR_HASH
    edited = vendor / "patch_maker.py"
    edited.write_bytes(edited.read_bytes() + b"\n")
    assert lwm_adapter.vendor_hash() != LWM.VENDOR_HASH


# --- weights -------------------------------------------------------------------------------


def test_weights_are_cached_at_the_pinned_revision(model):
    cached = ROOT / ".cache/weights/models--wi-lab--lwm-v1.1/snapshots" / SHA / "models/model.pth"
    assert cached.is_file()


def test_strict_load_has_no_missing_or_unexpected_keys(model):
    cached = ROOT / ".cache/weights/models--wi-lab--lwm-v1.1/snapshots" / SHA / "models/model.pth"
    state = strip_module_prefix(torch.load(cached, map_location="cpu"))
    with torch.device("meta"):
        expected = set(lwm().state_dict())
    assert set(state) == expected
    assert not any(p.is_meta for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == pytest.approx(2.4e6, rel=0.1)


# --- contract ------------------------------------------------------------------------------


@pytest.mark.parametrize("n_antennas", [4, 2])
@pytest.mark.parametrize("embedding", ["cls", "channel_mean"])
def test_packet_contract(make, n_antennas, embedding):
    extractor = make(embedding=embedding, batch_size=4)
    csi = _csi(n_antennas)
    check_packet_extractor(extractor, csi)
    out = extractor.extract_packets(csi)
    assert out.values.shape == (6, 128)
    assert out.feature_names == tuple(f"lwm.{embedding}.{i}" for i in range(128))
    assert set(out.input_stats) == {"median_abs", "mean_abs", "p5_abs", "p95_abs"}


def test_channel_mean_differs_from_cls(make):
    csi = _csi()
    cls = make(embedding="cls").extract_packets(csi).values
    mean = make(embedding="channel_mean").extract_packets(csi).values
    assert cls.shape == mean.shape == (6, 128)
    assert not np.allclose(cls, mean)


def test_batch_size_does_not_change_the_result(make):
    csi = _csi(n_packets=7)
    one = make(batch_size=512).extract_packets(csi).values
    small = make(batch_size=3).extract_packets(csi).values
    np.testing.assert_allclose(one, small, rtol=0, atol=1e-5)
    assert "batch_size" not in registry.affecting_params(LWM.Params(batch_size=3))


def test_one_antenna_is_padded_like_the_others(make):
    assert make().extract_packets(_csi(1)).values.shape == (6, 128)


def test_tokens_and_pooling_follow_the_upstream_layout(make, model):
    csi = _csi(n_antennas=2, n_packets=3)
    patches = patch_maker(csi.transpose(1, 0, 2)[:, None], 4, 4)
    # first patch of packet 0: antennas padded to 4 rows, real/imag interleaved per subcarrier
    first = np.zeros((4, 8), dtype=np.float32)
    first[:2, 0::2] = csi[:, 0, :4].real
    first[:2, 1::2] = csi[:, 0, :4].imag
    np.testing.assert_array_equal(patches[0, 0], first.ravel())
    cls = np.full((3, 1, 32), 0.2, dtype=np.float32)
    tokens = torch.from_numpy(np.concatenate([cls, patches], axis=1))
    assert tokens.shape == (3, 60, 32)
    with torch.inference_mode():
        output, _ = model(tokens)
    np.testing.assert_allclose(
        make(embedding="cls").extract_packets(csi).values, output[:, 0].numpy(), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        make(embedding="channel_mean").extract_packets(csi).values,
        output[:, 1:].mean(dim=1).numpy(),
        rtol=0,
        atol=1e-6,
    )


# --- frozen, no global state ---------------------------------------------------------------


def test_extraction_leaves_the_model_frozen(make, model):
    before = _checksum(model)
    make().extract_packets(_csi())
    make(embedding="channel_mean").extract_packets(_csi(2))
    assert _checksum(model) == before
    assert model.training is False
    assert all(not m.training for m in model.modules())


def test_global_rngs_are_untouched(monkeypatch):
    monkeypatch.chdir(ROOT)
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    extractor = LWM()  # lazy model load included
    extractor.extract_packets(_csi())
    assert extractor._model is not None
    after = np.random.get_state()
    assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1])
    assert np_state[2:] == after[2:]
    assert torch.equal(torch_state, torch.get_rng_state())


# --- amplitude domain check ----------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e12, 1e-3], ids=["raw", "tiny"])
def test_off_scale_input_warns_and_reports_the_median(make, caplog, scale):
    csi = _csi(scale=scale)
    with caplog.at_level(logging.WARNING, logger=lwm_adapter.__name__):
        out = make().extract_packets(csi)
    assert out.values.shape == (6, 128) and np.isfinite(out.values).all()
    median = out.input_stats["median_abs"]
    assert scale / 10 < median < scale * 10
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert f"{median:.6g}" in warnings[0].getMessage()


def test_order_one_input_does_not_warn(make, caplog):
    with caplog.at_level(logging.WARNING, logger=lwm_adapter.__name__):
        out = make().extract_packets(_csi())
    assert 0.1 <= out.input_stats["median_abs"] <= 10
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- bad input -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "csi",
    [
        _csi()[0],  # 2-D
        _csi()[:, :0],  # no packets
        np.concatenate([_csi(), _csi(1)]),  # 5 antennas
        np.abs(_csi()),  # real
        [[[1 + 1j]]],  # not an array
        _csi()[:, :, :0],  # no subcarriers
        np.ones((4, 2, 2049), dtype=np.complex64),  # 513 patches > 512 positions
        np.full((4, 2, 234), complex(np.inf, 0), dtype=np.complex64),  # non-finite
    ],
    ids=["2d", "no_packets", "five_antennas", "real", "list", "no_subcarriers", "too_long", "inf"],
)
def test_bad_input_raises(csi):
    extractor = LWM()
    with pytest.raises(ExtractorError):
        extractor.extract_packets(csi)
    assert extractor._model is None  # rejected before any model load


# --- config --------------------------------------------------------------------------------


LWM_BASE = {**BASE, "extractors": [{"name": "lwm"}], "pooler": "mean"}


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        (
            {"extractors": [{"name": "amplitude_stats"}, {"name": "lwm"}]},
            r"^extractors: per_packet",
        ),
        ({"pooler": None}, r"^pooler: required for per_packet"),
        ({"extractors": [{"name": "amplitude_stats"}]}, r"^pooler: must be null"),
        (
            {"pooler": "mean_std", "window_len": 1},
            r"^pooler\.name: window_len 1 < min_window_len 2",
        ),
        ({"pooler": "max"}, r"^pooler\.name: unknown pooler 'max'"),
    ],
    ids=["mixed", "missing_pooler", "pooler_on_per_window", "mean_std_w1", "unknown_pooler"],
)
def test_pooling_config_errors(discovered, changes, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict({**LWM_BASE, **changes})


def test_window_len_1_only_for_per_packet(discovered):
    assert config_from_dict({**LWM_BASE, "window_len": 1}).window_len == 1
    with pytest.raises(ConfigError, match=r"^window_len: 1 is valid only for per_packet"):
        config_from_dict({**BASE, "window_len": 1})  # amplitude_stats: min_window_len is 1


def test_lwm_config_defaults(discovered):
    config = config_from_dict(LWM_BASE)
    assert config.feature_selection is None
    assert config.pooler is not None and config.pooler.name == "mean"
    explicit = config_from_dict({**LWM_BASE, "pooler": {"name": "mean", "params": {}}})
    assert compute_run_id(explicit) == compute_run_id(config)


def test_pooler_enters_the_run_id(discovered):
    mean = config_from_dict(LWM_BASE)
    assert resolved_body(mean)["pooler"] == {"name": "mean", "version": 1, "params": {}}
    assert compute_run_id(mean) != compute_run_id(
        config_from_dict({**LWM_BASE, "pooler": "mean_std"})
    )
    assert "pooler" not in resolved_body(config_from_dict(BASE))


def test_existing_run_ids_are_unchanged(discovered):
    assert compute_run_id(config_from_dict(BASE)) == "39d3a969fb47"
    both = {**BASE, "extractors": [{"name": "amplitude_stats"}, {"name": "paper_beamforming"}]}
    assert compute_run_id(config_from_dict(both)) == "2105c1ec2677"
    body = resolved_body(config_from_dict(both))
    assert all("vendor_hash" not in e and "weights_revision" not in e for e in body["extractors"])


def test_run_id_body_carries_lwm_provenance(discovered):
    config = config_from_dict(LWM_BASE)
    (entry,) = resolved_body(config)["extractors"]
    assert entry["vendor_hash"] == LWM.VENDOR_HASH
    assert entry["weights_revision"] == SHA
    assert entry["params"] == {"embedding": "cls"}


# --- full dataset --------------------------------------------------------------------------


@pytest.mark.slow
def test_full_dataset_embeds_every_packet(caplog, monkeypatch):
    if not (RECORDINGS / "r1_empty.npy").is_file():
        pytest.skip("recordings not available")
    from csi_bench.adapters.datasets.tor_vergata import TorVergata
    from csi_bench.adapters.preprocess.global_rescale import GlobalRescale

    monkeypatch.chdir(ROOT)
    recordings = TorVergata(TorVergata.Params(root=str(RECORDINGS))).load()
    rescale = GlobalRescale()
    state = rescale.fit(recordings)  # fit on all recordings: fine for a timing check
    extractor = LWM()
    total = 0
    start = time.perf_counter()
    for rec in recordings:
        csi = rescale.transform(rec, state).csi
        with caplog.at_level(logging.WARNING, logger=lwm_adapter.__name__):
            out = extractor.extract_packets(csi)
        assert out.values.shape == (rec.meta.n_packets, 128)
        assert out.values.dtype == np.float32 and np.isfinite(out.values).all()
        total += rec.meta.n_packets
        logger.info(
            "%s: %d packets, input stats %s",
            rec.recording_id,
            len(out.values),
            {k: round(v, 4) for k, v in out.input_stats.items()},
        )
    elapsed = time.perf_counter() - start
    # global_rescale brings every recording to order 1, so the amplitude check stays quiet
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    logger.info(
        "lwm embedded %d packets of %d recordings in %.1f s", total, len(recordings), elapsed
    )
