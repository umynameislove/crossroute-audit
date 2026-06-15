"""Generic JSON-Schema loading and validation for CrossRoute-Audit artifacts."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:  # validation is optional for minimal installs
    Draft202012Validator = None


_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
_KNOWN = {"manifest", "control_status", "causal_effect", "audit_report"}


def load_schema(name: str) -> dict:
    """Load schemas/<name>.schema.json for a known artifact type."""
    if name not in _KNOWN:
        raise ValueError(
            f"unknown schema name {name!r}; expected one of {sorted(_KNOWN)}"
        )
    path = _SCHEMA_DIR / f"{name}.schema.json"
    if not path.is_file():
        raise FileNotFoundError(f"schema file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifact(obj: dict, name: str) -> None:
    """Raise ValueError if an artifact violates its JSON Schema.

    Validation is a no-op when jsonschema is unavailable, keeping the package usable
    in minimal non-validation environments.
    """
    if Draft202012Validator is None:
        return

    validator = Draft202012Validator(load_schema(name))
    errors = sorted(
        validator.iter_errors(obj),
        key=lambda error: (list(error.path), error.message),
    )
    if errors:
        joined = "; ".join(_format_error(error) for error in errors)
        raise ValueError(f"{name} schema validation failed: {joined}")


def _format_error(error) -> str:
    path = "$"
    for part in error.path:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return f"{path}: {error.message}"
