import sys

import pytest
import yaml

from csi_bench.config import ExperimentBuilder
from csi_bench.config.experiment import ExperimentConfig, load_config
from csi_bench.core import registry
from csi_bench.core.errors import ConfigError
from csi_bench.runner.discover import discover


@pytest.fixture(autouse=True)
def discovered(monkeypatch):
    registry.reset()
    for name in [m for m in sys.modules if m.startswith("csi_bench.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    discover()
    yield
    registry.reset()


def _builder() -> ExperimentBuilder:
    return (
        ExperimentBuilder()
        .dataset("tor_vergata")
        .rooms("r1")
        .preprocessing("ant2")
        .window(40)
        .features("amplitude_stats", ("paper_beamforming", {}))
        .feature_selection({"mrmr": {"k": [2, 3]}})
        .head("svm")
        .seeds(0, 1)
        .role("comparison")
    )


def test_builder_equals_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": "tor_vergata",
                "selection": {"rooms": ["r1"]},
                "preprocessing": ["ant2"],
                "window_len": 40,
                "extractors": [{"name": "amplitude_stats"}, {"name": "paper_beamforming"}],
                "feature_selection": {"mrmr": {"k": [2, 3]}},
                "head": {"name": "svm"},
                "seeds": [0, 1],
                "role": "comparison",
            }
        )
    )
    built = _builder().build()
    assert isinstance(built, ExperimentConfig)
    assert built == load_config(path)


def test_missing_required_field():
    with pytest.raises(ConfigError, match="head"):
        ExperimentBuilder().dataset("tor_vergata").rooms("r1").window(40).features(
            "amplitude_stats"
        ).role("comparison").build()


def test_build_touches_no_dataset(monkeypatch):
    from csi_bench.adapters.datasets.tor_vergata import TorVergata

    def boom(*a, **k):
        raise AssertionError("dataset touched")

    monkeypatch.setattr(TorVergata, "load", boom)
    monkeypatch.setattr(TorVergata, "__init__", boom)
    _builder().build()
