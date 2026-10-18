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
    # LWM input variants (antenna-agnostic; list ant2/ant4 first, e.g. [ant2, lwm_power])
    "lwm_rescale": [{"name": "global_rescale", "params": {}}],
    "lwm_power": [{"name": "per_packet_power", "params": {}}],
    "lwm_rescale_phase": [
        {"name": "global_rescale", "params": {}},
        {"name": "phase_sanitize", "params": {}},
    ],
    "lwm_power_phase": [
        {"name": "per_packet_power", "params": {}},
        {"name": "phase_sanitize", "params": {}},
    ],
    "lwm_amplitude": [
        {"name": "global_rescale", "params": {}},
        {"name": "amplitude_only", "params": {}},
    ],
}
