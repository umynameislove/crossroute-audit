"""Load and validate the sample manifest against the JSON schema."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:  # validation is optional but recommended
    Draft202012Validator = None

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "manifest.schema.json"


def load_schema() -> dict:
    """Return the manifest JSON schema as a dict."""
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def load_manifest(path: str | Path) -> list[dict]:
    """Read a JSONL manifest into a list of records, validating each line."""
    records: list[dict] = []
    validator = Draft202012Validator(load_schema()) if Draft202012Validator else None
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if validator is not None:
            errors = sorted(validator.iter_errors(record), key=lambda e: list(e.path))
            if errors:
                raise ValueError(
                    f"manifest line {line_no}: " + "; ".join(e.message for e in errors)
                )
        records.append(record)
    return records


if __name__ == "__main__":
    import sys

    loaded = load_manifest(sys.argv[1])
    print(f"OK: {len(loaded)} samples validated")
