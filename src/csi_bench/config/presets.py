"""Named preprocessing presets; expanded in place when the config loads."""

from typing import Any

PREPROCESS_PRESETS: dict[str, list[dict[str, Any]]] = {
    "ant2": [{"name": "antennas", "params": {"ids": [0, 1]}}],
    "ant4": [{"name": "antennas", "params": {"ids": [0, 1, 2, 3]}}],
    "ant2_rescale": [
        {"name": "antennas", "params": {"ids": [0, 1]}},
        {"name": "global_rescale", "params": {}},
    ],
    "ant4_rescale": [
        {"name": "antennas", "params": {"ids": [0, 1, 2, 3]}},
        {"name": "global_rescale", "params": {}},
    ],
    "paper_norm": [{"name": "recording_mean_norm", "params": {}}],
}
