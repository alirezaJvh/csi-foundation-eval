"""Frozen LWM v1.1 per-packet embeddings (Story 5.1).

Each packet `(antennas, subcarriers)` becomes 4x4 complex patches (real/imag interleaved, rows
zero-padded to 4 by the vendored `patch_maker`), a CLS token is prepended and the frozen model's
CLS output or the mean of its patch outputs is returned. Pooling and caching belong to the runner.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field

from csi_bench.adapters.extractors.lwm._vendor.lwm_model import lwm
from csi_bench.adapters.extractors.lwm._vendor.patch_maker import patch_maker
from csi_bench.core.contracts import PacketFeatures
from csi_bench.core.errors import ExtractorError
from csi_bench.core.registry import register

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
VENDOR_FILES = ("SOURCE.md", "__init__.py", "lwm_model.py", "patch_maker.py")
REPO_ID = "wi-lab/lwm-v1.1"
WEIGHTS_FILE = "models/model.pth"
WEIGHTS_CACHE = ".cache/weights"
REVISION = "1b68512dbe97de56841301206eacb780fcd76011"
PATCH_ROWS = PATCH_COLS = 4
ELEMENT_LENGTH = PATCH_ROWS * PATCH_COLS * 2  # 32: real and imaginary parts
D_MODEL = 128
MAX_LEN = 513  # positional table size, CLS included
MAX_ANTENNAS = PATCH_ROWS
CLS_VALUE = 0.2  # upstream `tokenizer`: '[CLS]': 0.2 * np.ones(patch_size)
AMPLITUDE_RANGE = (0.1, 10.0)


def vendor_hash() -> str:
    """sha256 over the vendored files (name and bytes, in a fixed order)."""
    digest = hashlib.sha256()
    for name in VENDOR_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((VENDOR_DIR / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """Drop the `module.` prefix that `nn.DataParallel` adds to checkpoint keys."""
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_model(weights_path: str | Path | None = None) -> torch.nn.Module:
    """Build `lwm()` with its defaults and load the pinned weights strictly, in eval mode.

    The model is built on the meta device and its tensors are assigned from the checkpoint,
    so construction never draws from the global torch RNG.
    """
    if weights_path is None:
        weights_path = hf_hub_download(
            REPO_ID, WEIGHTS_FILE, revision=REVISION, cache_dir=WEIGHTS_CACHE
        )
    state = torch.load(weights_path, map_location="cpu")
    with torch.device("meta"):
        model = lwm()
    model.load_state_dict(strip_module_prefix(state), strict=True, assign=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def input_stats(csi: np.ndarray) -> dict[str, float]:
    amplitude = np.abs(csi).astype(np.float64).ravel()
    p5, median, p95 = np.percentile(amplitude, [5, 50, 95])
    return {
        "median_abs": float(median),
        "mean_abs": float(amplitude.mean()),
        "p5_abs": float(p5),
        "p95_abs": float(p95),
    }


def _validate(csi: Any) -> np.ndarray:
    if not isinstance(csi, np.ndarray):
        raise ExtractorError(f"lwm expects a numpy array, got {type(csi).__name__}")
    if csi.ndim != 3:
        raise ExtractorError(f"lwm expects (antennas, packets, subcarriers), got shape {csi.shape}")
    if not np.iscomplexobj(csi):
        raise ExtractorError(f"lwm expects complex CSI, got dtype {csi.dtype}")
    n_antennas, n_packets, n_subcarriers = csi.shape
    if not 1 <= n_antennas <= MAX_ANTENNAS:
        raise ExtractorError(f"lwm takes 1-{MAX_ANTENNAS} antennas, got {n_antennas}")
    if n_packets == 0:
        raise ExtractorError("lwm received 0 packets")
    n_tokens = -(-n_subcarriers // PATCH_COLS)
    if not 1 <= n_tokens <= MAX_LEN - 1:
        raise ExtractorError(
            f"lwm: {n_subcarriers} subcarriers give {n_tokens} patches; "
            f"1-{MAX_LEN - 1} are supported"
        )
    if not np.isfinite(csi).all():
        raise ExtractorError("lwm received non-finite CSI")
    return csi.astype(np.complex64, copy=False)


@register("extractor", "lwm")
class LWM:
    VERSION = 1
    granularity = "per_packet"
    requires = {"phase": True, "n_antennas": "any"}
    min_window_len = 1
    VENDOR_HASH = vendor_hash()
    WEIGHTS_REVISION = REVISION

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        embedding: Literal["cls", "channel_mean"] = "cls"
        batch_size: int = Field(default=512, ge=1, json_schema_extra={"affects_result": False})

    def __init__(self, params: "LWM.Params | None" = None) -> None:
        self.params = params or self.Params()
        self._model: torch.nn.Module | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"lwm.{self.params.embedding}.{i}" for i in range(D_MODEL))

    def _get_model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = load_model()
        return self._model

    def _embed(self, model: torch.nn.Module, packets: np.ndarray) -> np.ndarray:
        """`packets`: complex64 `(n, antennas, subcarriers)` -> float32 `(n, 128)`."""
        patches = patch_maker(packets[:, None], PATCH_ROWS, PATCH_COLS)
        cls = np.full((len(patches), 1, ELEMENT_LENGTH), CLS_VALUE, dtype=np.float32)
        tokens = torch.from_numpy(np.concatenate([cls, patches], axis=1))
        output, _ = model(tokens)
        if self.params.embedding == "cls":
            pooled = output[:, 0]
        else:
            pooled = output[:, 1:].mean(dim=1)
        return pooled.numpy().astype(np.float32, copy=False)

    def extract_packets(self, csi: np.ndarray) -> PacketFeatures:
        csi = _validate(csi)
        stats = input_stats(csi)
        low, high = AMPLITUDE_RANGE
        if not low <= stats["median_abs"] <= high:
            logger.warning(
                "lwm: input median |CSI| = %.6g is outside [%g, %g]; LWM was pretrained on "
                "order-1 channels, rescale the input",
                stats["median_abs"],
                low,
                high,
            )
        model = self._get_model()
        packets = csi.transpose(1, 0, 2)
        size = self.params.batch_size
        rows = []
        with torch.inference_mode():
            for start in range(0, len(packets), size):
                rows.append(self._embed(model, packets[start : start + size]))
        values = np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)
        if not np.isfinite(values).all():
            raise ExtractorError("lwm produced non-finite embeddings")
        return PacketFeatures(values=values, feature_names=self.feature_names, input_stats=stats)
