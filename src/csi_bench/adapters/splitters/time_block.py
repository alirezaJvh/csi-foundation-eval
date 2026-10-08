"""Contiguous train | gap | val | gap | test packet blocks per recording."""

import dataclasses
import math
from collections import defaultdict

from pydantic import BaseModel

from csi_bench.core.contracts import Recording, SplitPlan
from csi_bench.core.errors import ConfigError, DataError
from csi_bench.core.registry import register
from csi_bench.core.splitplan import compute_split_hash


@register("splitter", "time_block")
class TimeBlock:
    VERSION = 1

    class Params(BaseModel):
        gap_packets: int = 64
        fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)

    def __init__(self, params: "TimeBlock.Params | None" = None) -> None:
        self.params = params or self.Params()
        p = self.params
        if p.gap_packets < 0:
            raise ConfigError("gap_packets must be >= 0")
        if any(f <= 0 for f in p.fractions) or not math.isclose(sum(p.fractions), 1.0):
            raise ConfigError(f"fractions must be > 0 and sum to 1, got {p.fractions}")

    def split(self, recordings: list[Recording], dataset: str) -> SplitPlan:
        gap = self.params.gap_packets
        _, f_val, f_test = self.params.fractions
        blocks: dict[str, dict[str, tuple[int, int]]] = {}
        sizes: dict[int, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
        for rec in sorted(recordings, key=lambda r: r.recording_id):
            n = rec.meta.n_packets
            usable = n - 2 * gap
            if usable <= 0:
                raise DataError(
                    f"recording {rec.recording_id} too short for the split: "
                    f"{n} packets, gap_packets={gap}"
                )
            val = math.floor(f_val * usable)
            test = math.floor(f_test * usable)
            train = usable - val - test
            v0 = train + gap
            t0 = v0 + val + gap
            blocks[rec.recording_id] = {
                "train": (0, train),
                "val": (v0, v0 + val),
                "test": (t0, t0 + test),
            }
            for part, size in (("train", train), ("val", val), ("test", test)):
                sizes[rec.label][part] += size
        for label, parts in sorted(sizes.items()):
            for part, size in parts.items():
                if size <= 0:
                    raise DataError(f"class {label} has no {part} packets")
        plan = SplitPlan(
            dataset=dataset,
            splitter="time_block",
            params=self.params.model_dump(mode="json"),
            gap_packets=gap,
            blocks=blocks,
            source_sha256={r.recording_id: r.meta.source_sha256 for r in recordings},
            split_hash="",
        )
        return dataclasses.replace(plan, split_hash=compute_split_hash(plan))
