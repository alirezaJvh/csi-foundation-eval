"""Error types shared across the project."""


class ConfigError(Exception):
    """Invalid or inconsistent configuration, registry misuse."""


class DataError(Exception):
    """Dataset or split problem."""


class ExtractorError(Exception):
    """Feature extractor failure."""
