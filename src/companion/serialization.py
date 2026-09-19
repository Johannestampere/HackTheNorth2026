"""Small, explicit versioned JSON envelope shared by file adapters."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def read_document(path: Path, expected_kind: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"schema_version", "kind", "data"}:
        raise ValueError("Expected an envelope containing schema_version, kind, and data")
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {document['schema_version']!r}")
    if document["kind"] != expected_kind or not isinstance(document["data"], dict):
        raise ValueError(f"Expected document kind {expected_kind!r} with object data")
    return document["data"]


def write_document(path: Path, kind: str, value: Any) -> None:
    document = {"schema_version": SCHEMA_VERSION, "kind": kind, "data": asdict(value)}
    payload = json.dumps(document, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
