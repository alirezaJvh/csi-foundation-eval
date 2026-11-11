"""Cached dataset loads hold no file descriptor per recording (Story 6.5).

Each load runs in a subprocess whose `RLIMIT_NOFILE` is lowered below the number of
recordings, as parallel sweeps with forked workers do to the shared fd budget.
"""

import errno
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from csi_bench.core.errors import DataError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs RLIMIT_NOFILE")

ROOT = Path(__file__).resolve().parents[1]
N_RECORDINGS = 100
LOW_LIMIT = 64

_SCRIPT = """
import hashlib, importlib, json, os, resource, sys
import numpy as np

module, cls_name, root, limit = sys.argv[1:5]
cls = getattr(importlib.import_module(module), cls_name)

def fds():
    return len(os.listdir("/dev/fd")) - 1  # minus the directory listing's own fd

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
before = fds()
resource.setrlimit(resource.RLIMIT_NOFILE, (int(limit), hard))
try:
    recs = cls(cls.Params(root=root)).load()
except Exception as e:
    print(json.dumps({"error": type(e).__name__, "message": str(e)}))
    sys.exit(1)
after = fds()
digest = hashlib.sha256()
for r in recs:
    digest.update(r.recording_id.encode())
    digest.update(np.ascontiguousarray(r.csi).tobytes())
print(json.dumps({"n": len(recs), "before": before, "after": after, "sha": digest.hexdigest()}))
"""

_DATASETS = {
    "csi_bench_hp": ("csi_bench.adapters.datasets.csi_bench_hp", "CsiBenchHp"),
    "padova": ("csi_bench.adapters.datasets.padova", "Padova"),
}


def _write_csi_bench_hp(root: Path, n: int) -> None:
    rng = np.random.default_rng(0)
    d = root / "user_U01" / "act_jumping" / "env_E01" / "device_HP"
    d.mkdir(parents=True)
    for i in range(n):
        shape = (2, 2, 58, 4)
        csi = rng.normal(size=shape) + 1j * rng.normal(size=shape) + 5
        savemat(d / f"rec_{i:03d}.mat", {"csi_trace": {"csi": csi}})


def _write_padova(root: Path, n: int) -> None:
    rng = np.random.default_rng(0)
    d = root / "S1a"
    d.mkdir(parents=True)
    for i in range(n):
        rows = rng.normal(size=(8, 256)) + 1j * rng.normal(size=(8, 256)) + 5
        savemat(d / f"S1a_E{i + 1}.mat", {"csi_buff": rows})


_WRITERS = {"csi_bench_hp": _write_csi_bench_hp, "padova": _write_padova}


def _start(dataset: str, root: Path, limit: int | str) -> subprocess.Popen:
    module, cls_name = _DATASETS[dataset]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    return subprocess.Popen(
        [sys.executable, "-c", _SCRIPT, module, cls_name, str(root), str(limit)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=ROOT,
    )


def _result(proc: subprocess.Popen) -> dict:
    out, err = proc.communicate(timeout=300)
    assert out.strip(), err
    return json.loads(out.strip().splitlines()[-1])


@pytest.fixture(params=sorted(_DATASETS))
def dataset(request):
    return request.param


def test_cached_load_under_low_fd_limit(tmp_path, dataset):
    _WRITERS[dataset](tmp_path, N_RECORDINGS)
    warm = _result(_start(dataset, tmp_path, "4096"))  # fill the cache
    assert warm.get("n") == N_RECORDINGS, warm
    result = _result(_start(dataset, tmp_path, LOW_LIMIT))
    assert result.get("n") == N_RECORDINGS, result
    assert result["after"] - result["before"] <= 2, result
    assert result["sha"] == warm["sha"]


def test_concurrent_cold_loads_under_low_fd_limit(tmp_path, dataset):
    _WRITERS[dataset](tmp_path, N_RECORDINGS)
    procs = [_start(dataset, tmp_path, LOW_LIMIT) for _ in range(2)]
    first, second = (_result(p) for p in procs)
    assert first.get("n") == N_RECORDINGS, first
    assert second.get("n") == N_RECORDINGS, second
    assert first["sha"] == second["sha"]


@pytest.mark.parametrize("code", [errno.EMFILE, errno.ENFILE, errno.ENOMEM])
def test_exhausted_fds_name_the_cause(tmp_path, monkeypatch, dataset, code):
    """No fd or mapping free when a cached array is opened: a DataError naming the cause,
    not a reconversion reported as "cache file unreadable"."""
    _WRITERS[dataset](tmp_path, 3)
    module_name, cls_name = _DATASETS[dataset]
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    assert len(cls(cls.Params(root=str(tmp_path))).load()) == 3

    def exhausted(path):
        raise OSError(code, os.strerror(code), str(path))

    monkeypatch.setattr(module, "map_npy", exhausted)
    with pytest.raises(DataError, match="file descriptors, memory mappings or memory"):
        cls(cls.Params(root=str(tmp_path))).load()
