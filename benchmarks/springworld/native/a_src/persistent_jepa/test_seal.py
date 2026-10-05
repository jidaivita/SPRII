"""Immutable selection record and guarded test access."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


class TestSealedError(RuntimeError):
    pass


def require_test_unsealed(selection_path: Path) -> dict:
    if not selection_path.is_file():
        raise TestSealedError("test access denied: immutable selection.json is absent")
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    required = {
        "variant",
        "full_config",
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "primary_metric",
        "tie_breakers",
        "decoder_ridge",
        "pairing_seed",
        "code_revision",
        "created_at",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise TestSealedError(f"test access denied: selection.json missing {missing}")
    return payload


def write_immutable_selection(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"selection is immutable and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_keys = payload.copy()
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(require_keys, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    path.chmod(0o444)

