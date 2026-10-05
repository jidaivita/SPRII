"""Validate the Python-generated FHN MAT bank against the released loader contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from scipy import io as sio


TRAIN = ((0.03, 0.10), (0.02, 0.15), (0.04, 0.15),
         (0.01, 0.20), (0.03, 0.20), (0.05, 0.20),
         (0.02, 0.25), (0.04, 0.25), (0.03, 0.30))
OOD_INTRA = ((0.03, 0.15), (0.02, 0.20), (0.04, 0.20), (0.03, 0.25))
OOD_EXTRA = ((0.01, 0.10), (0.02, 0.10), (0.04, 0.10), (0.05, 0.10),
             (0.01, 0.15), (0.05, 0.15), (0.01, 0.25), (0.05, 0.25),
             (0.01, 0.30), (0.02, 0.30), (0.04, 0.30), (0.05, 0.30))
EVAL_IDS = (5, 15, 24, 36, 45)
TRAIN_IDS = tuple(range(50, 90))
NAME_RE = re.compile(r"^(U|V)_(0\.0[1-5])_(0\.[123][05])_(\d+)\.mat$")


def expected_names() -> set[str]:
    specs = [(TRAIN, TRAIN_IDS), (TRAIN, EVAL_IDS),
             (OOD_INTRA, EVAL_IDS), (OOD_EXTRA, EVAL_IDS)]
    return {
        f"{field}_{k:.2f}_{beta:.2f}_{initial_id}.mat"
        for systems, ids in specs
        for k, beta in systems
        for initial_id in ids
        for field in ("U", "V")
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--hash-all", action="store_true")
    args = parser.parse_args()

    start = time.time()
    files = sorted(args.data_dir.glob("*.mat"))
    actual = {path.name for path in files}
    expected = expected_names()
    errors: list[str] = []
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        errors.append(f"missing={missing[:20]} count={len(missing)}")
        errors.append(f"extra={extra[:20]} count={len(extra)}")

    ranges = {"U": [math.inf, -math.inf], "V": [math.inf, -math.inf]}
    hashes: dict[str, str] = {}
    for index, path in enumerate(files, 1):
        match = NAME_RE.match(path.name)
        if not match:
            errors.append(f"bad filename: {path.name}")
            continue
        field = match.group(1)
        variable = f"{field}_record"
        loaded = sio.loadmat(path)
        if variable not in loaded:
            errors.append(f"{path.name}: missing {variable}")
            continue
        array = loaded[variable]
        if array.shape != (101, 128, 128):
            errors.append(f"{path.name}: shape={array.shape}")
        if array.dtype != np.float32:
            errors.append(f"{path.name}: dtype={array.dtype}")
        if not np.isfinite(array).all():
            errors.append(f"{path.name}: non-finite values")
        ranges[field][0] = min(ranges[field][0], float(array.min()))
        ranges[field][1] = max(ranges[field][1], float(array.max()))
        if args.hash_all:
            hashes[path.name] = sha256(path)
        elif index in (1, len(files) // 2, len(files)):
            hashes[path.name] = sha256(path)

    receipt = {
        "status": "PASS" if not errors else "FAIL",
        "data_dir": str(args.data_dir),
        "expected_file_count": len(expected),
        "actual_file_count": len(files),
        "shape": [101, 128, 128],
        "dtype": "float32",
        "finite": not any("non-finite" in err for err in errors),
        "value_ranges": ranges,
        "sample_or_all_sha256": hashes,
        "hash_all": args.hash_all,
        "errors": errors,
        "elapsed_seconds": time.time() - start,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
