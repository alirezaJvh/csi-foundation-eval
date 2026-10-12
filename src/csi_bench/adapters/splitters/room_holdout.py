"""Cross-room split: train | gap | val blocks in the train rooms, whole test-room recordings."""

import dataclasses
import math
from collections import defaultdict

from pydantic import BaseModel

from csi_bench.core.contracts import Recording, SplitPlan
from csi_bench.core.errors import ConfigError, DataError
from csi_bench.core.registry import register
from csi_bench.core.splitplan import PARTS, ROOM_HOLDOUT, compute_split_hash

EMPTY = (0, 0)  # an empty part


@register("splitter", ROOM_HOLDOUT)
class RoomHoldout:
    VERSION = 1

    class Params(BaseModel):
        train_rooms: list[str]
        test_rooms: list[str]
        fractions: tuple[float, float] = (0.75, 0.25)
        gap_packets: int = 64

    def __init__(self, params: "RoomHoldout.Params") -> None:
        self.params = params
        p = self.params
        if not p.train_rooms or not p.test_rooms:
            raise ConfigError("train_rooms and test_rooms must be non-empty")
        shared = sorted(set(p.train_rooms) & set(p.test_rooms))
        if shared:
            raise ConfigError(f"train_rooms and test_rooms must be disjoint, both have {shared}")
        if p.gap_packets < 0:
            raise ConfigError("gap_packets must be >= 0")
        if any(f <= 0 for f in p.fractions) or not math.isclose(sum(p.fractions), 1.0):
            raise ConfigError(f"fractions must be > 0 and sum to 1, got {p.fractions}")

    def split(self, recordings: list[Recording], dataset: str) -> SplitPlan:
        p = self.params
        gap = p.gap_packets
        _, f_val = p.fractions
        train_rooms, test_rooms = set(p.train_rooms), set(p.test_rooms)
        missing = sorted((train_rooms | test_rooms) - {r.room for r in recordings})
        if missing:
            known = sorted({r.room for r in recordings})
            raise ConfigError(f"rooms {missing} have no recordings; dataset rooms: {known}")
        blocks: dict[str, dict[str, tuple[int, int]]] = {}
        sizes: dict[int, dict[str, int]] = defaultdict(lambda: dict.fromkeys(PARTS, 0))
        for rec in sorted(recordings, key=lambda r: r.recording_id):
            n = rec.meta.n_packets
            if rec.room in train_rooms:
                usable = n - gap
                val = math.floor(f_val * usable)
                train = usable - val
                if usable <= 0 or val <= 0 or train <= 0:
                    raise DataError(
                        f"recording {rec.recording_id} too short for the split: "
                        f"{n} packets, gap_packets={gap}"
                    )
                v0 = train + gap
                blocks[rec.recording_id] = {
                    "train": (0, train),
                    "val": (v0, v0 + val),
                    "test": EMPTY,
                }
                sizes[rec.label]["train"] += train
                sizes[rec.label]["val"] += val
            elif rec.room in test_rooms:
                blocks[rec.recording_id] = {"train": EMPTY, "val": EMPTY, "test": (0, n)}
                sizes[rec.label]["test"] += n
            else:
                blocks[rec.recording_id] = dict.fromkeys(PARTS, EMPTY)
        labels = {r.label for r in recordings if r.room in train_rooms | test_rooms}
        for label in sorted(labels):
            for part in PARTS:
                if sizes[label][part] <= 0:
                    raise DataError(f"class {label} has no {part} packets")
        plan = SplitPlan(
            dataset=dataset,
            splitter=ROOM_HOLDOUT,
            params=p.model_dump(mode="json"),
            gap_packets=gap,
            blocks=blocks,
            source_sha256={r.recording_id: r.meta.source_sha256 for r in recordings},
            split_hash="",
        )
        return dataclasses.replace(plan, split_hash=compute_split_hash(plan))
