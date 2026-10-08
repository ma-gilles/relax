"""Environment flag parsing shared by dense and local EM engines."""

from __future__ import annotations

import logging
import os


def parse_env_true_flag(name: str) -> bool:
    """Enable recognized true tokens; absent or unrecognized values stay disabled."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def parse_env_flag_or_false(name: str, *, logger: logging.Logger) -> bool:
    """Read a recognized boolean token; invalid values warn and stay disabled.

    Unlike ``parse_env_flag``, an unknown non-empty value does not enable the
    flag. Unlike ``parse_env_binary_flag``, words and blank values are accepted.
    Read at call time and retain the caller's diagnostic logging context.
    """
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    logger.warning("Ignoring invalid %s=%r; using default false", name, value)
    return False


def parse_env_optional_flag(name: str, *, logger: logging.Logger, fallback: str) -> bool | None:
    """Read a boolean token; None when unset or blank, so the caller applies its default.

    An unrecognized value warns through the caller's logger, naming ``fallback`` (the default the caller
    then uses), and is None too.
    """
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Ignoring invalid %s=%r; using %s", name, value, fallback)
    return None


def parse_env_choice(name: str, choices: dict, *, logger: logging.Logger, expected: str):
    """Map a token through ``choices`` (keys lower case, ``-`` read as ``_``); None when unset or blank.

    An unrecognized value warns through the caller's logger with ``expected`` and is None.
    """
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower().replace("-", "_")
    if normalized in choices:
        return choices[normalized]
    logger.warning("Ignoring invalid %s=%r; expected %s", name, value, expected)
    return None


def parse_env_strict_flag(name: str, *, default: bool = False) -> bool:
    """Read a recognized boolean token, failing closed on anything else.

    The caller's default selects which token an unset variable behaves like.
    Unlike :func:`parse_env_flag_or_false`, an unrecognized value raises instead
    of quietly disabling the policy, so a misspelled opt-in cannot silently run
    the default path.
    """
    token = os.environ.get(name, "1" if default else "0").strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"Unsupported {name}={token!r}")


def parse_env_auto_flag(name: str, *, logger: logging.Logger) -> bool | None:
    """Read an ``auto``/forced switch: None for ``auto`` (also unset), True or False when forced.

    ``never`` and the false tokens force off, ``always`` and the true tokens force on; any other value
    warns through the caller's logger and is ``auto``. Whitespace and case are ignored.
    """
    mode = os.environ.get(name, "auto").strip().lower()
    if mode in {"0", "false", "no", "off", "never"}:
        return False
    if mode in {"1", "true", "yes", "on", "always"}:
        return True
    if mode != "auto":
        logger.warning("Unrecognised %s=%r; using auto", name, mode)
    return None


def parse_env_binary_flag(name: str) -> bool:
    """Read a strict 0/1 flag; unset is false, whitespace is stripped, blank is invalid."""
    token = os.environ.get(name, "0").strip()
    if token not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return token == "1"


def _parse_env_number_or_default(name, default, cast, default_format: str, *, logger: logging.Logger):
    """Read a numeric override with ``cast``; unset or blank uses ``default``, an invalid value warns and uses it."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return cast(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using " + default_format, name, value, default)
        return default


def parse_env_float_or_default(name: str, default: float, *, logger: logging.Logger) -> float:
    """Read a float override, warning through the caller's logger if invalid."""
    return _parse_env_number_or_default(name, default, float, "%.3f", logger=logger)


def parse_env_int_or_default(name: str, default: int, *, logger: logging.Logger) -> int:
    """Read an integer override, warning through the caller's logger if invalid."""
    return _parse_env_number_or_default(name, default, int, "%d", logger=logger)


def parse_int_set(value: str | None) -> set[int] | None:
    """Parse comma/semicolon/whitespace separated integer sets."""

    if not value:
        return None
    parsed = {int(token) for token in value.replace(",", " ").replace(";", " ").split()}
    return parsed or None


def parse_env_int_set(name: str) -> set[int] | None:
    """Parse an integer-set environment variable."""

    return parse_int_set(os.environ.get(name))


def parse_env_nonnegative_int(name: str) -> int | None:
    """Read a non-negative integer; unset or empty values have no override."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


def parse_env_flag(name: str, *, default: bool = False) -> bool:
    """Read a boolean override; unset or blank values use the caller default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def parse_env_capacity_ladder(name: str, default: tuple) -> tuple:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return tuple(int(v) for v in default)
    values = tuple(int(part) for part in raw.replace(",", " ").split())
    if not values or list(values) != sorted(values) or values[0] <= 0:
        raise ValueError(f"{name} must be an increasing list of positive integers, got {raw!r}")
    return values

