"""Paper-style random assignment of individual windows (reference runs only)."""

import dataclasses
import math
from collections import defaultdict

from pydantic import BaseModel

from csi_bench.core.contracts import Recording, SplitPlan
from csi_bench.core.errors import ConfigError, DataError
from csi_bench.core.registry import register
from csi_bench.core.splitplan import compute_split_hash


@register("splitter", "random_window")
class RandomWindow:
    VERSION = 1

    class Params(BaseModel):
        seed: int = 0
        fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)

    def __init__(self, params: "RandomWindow.Params | None" = None) -> None:
        self.params = params or self.Params()
        fractions = self.params.fractions
        if any(f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1.0):
            raise ConfigError(f"fractions must be > 0 and sum to 1, got {fractions}")

    def split(self, recordings: list[Recording], dataset: str) -> SplitPlan:
        """The plan holds seed and fractions; the Windower derives each window's part from them.

        Blocks span whole recordings, so one plan serves every window length and room set.
        Part sizes are estimated with `window_len` 1, the most windows any length can give.
        """
        _, f_val, f_test = self.params.fractions
        sizes: dict[int, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
        for rec in recordings:
            n = rec.meta.n_packets
            val = math.floor(f_val * n)
            test = math.floor(f_test * n)
            for part, size in (("train", n - val - test), ("val", val), ("test", test)):
                sizes[rec.label][part] += size
        for label, parts in sorted(sizes.items()):
            for part, size in parts.items():
                if size <= 0:
                    raise DataError(f"class {label} has no {part} windows")
        blocks = {
            rec.recording_id: {part: (0, rec.meta.n_packets) for part in ("train", "val", "test")}
            for rec in sorted(recordings, key=lambda r: r.recording_id)
        }
        plan = SplitPlan(
            dataset=dataset,
            splitter="random_window",
            params=self.params.model_dump(mode="json"),
            gap_packets=0,
            blocks=blocks,
            source_sha256={r.recording_id: r.meta.source_sha256 for r in recordings},
            split_hash="",
            unit="window",
        )
        return dataclasses.replace(plan, split_hash=compute_split_hash(plan))
