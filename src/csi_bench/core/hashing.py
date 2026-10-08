"""Canonical-JSON hashing."""

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize with sorted keys and no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def short_hash(obj: Any) -> str:
    """First 12 hex chars of the SHA-256 of the canonical JSON."""
    return sha256_hex(obj)[:12]
