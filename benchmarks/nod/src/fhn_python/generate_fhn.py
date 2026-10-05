#!/usr/bin/env python3
"""Generate loader-compatible NOD FHN/DR2D data without MATLAB."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .solver import (
        FHNConfig,
        gaussian_random_fields,
        iid_gaussian_initial_conditions,
        simulate,
        torch_simulate,
    )
except ImportError:  # direct ``python generate_fhn.py`` execution
    from solver import (
        FHNConfig,
        gaussian_random_fields,
        iid_gaussian_initial_conditions,
        simulate,
        torch_simulate,
    )


TRAIN = ((0.03, 0.10), (0.02, 0.15), (0.04, 0.15),
         (0.01, 0.20), (0.03, 0.20), (0.05, 0.20),
         (0.02, 0.25), (0.04, 0.25), (0.03, 0.30))
OOD_INTRA = ((0.03, 0.15), (0.02, 0.20), (0.04, 0.20), (0.03, 0.25))
OOD_EXTRA = ((0.01, 0.10), (0.02, 0.10), (0.04, 0.10), (0.05, 0.10),
             (0.01, 0.15), (0.05, 0.15), (0.01, 0.25), (0.05, 0.25),
             (0.01, 0.30), (0.02, 0.30), (0.04, 0.30), (0.05, 0.30))
SPLITS = {"train": TRAIN, "ood-intra": OOD_INTRA, "ood-extra": OOD_EXTRA}


def parse_ids(spec: str) -> list[int]:
    result: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(token))
    if not result or min(result) < 1:
        raise argparse.ArgumentTypeError("initial IDs must be positive and nonempty")
    return sorted(result)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_loader_files(
    output_dir: Path,
    records: np.ndarray,
    *,
    k: float,
    beta: float,
    initial_ids: list[int],
    compress: bool,
) -> list[dict[str, object]]:
    try:
        from scipy.io import savemat
    except ImportError as exc:
        raise RuntimeError("SciPy is required to write loader-compatible .mat files") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row, initial_id in enumerate(initial_ids):
        u_path = output_dir / f"U_{k:.2f}_{beta:.2f}_{initial_id:d}.mat"
        v_path = output_dir / f"V_{k:.2f}_{beta:.2f}_{initial_id:d}.mat"
        savemat(u_path, {"U_record": records[row, :, 0]}, do_compression=compress)
        savemat(v_path, {"V_record": records[row, :, 1]}, do_compression=compress)
        for path, variable in ((u_path, "U_record"), (v_path, "V_record")):
            written.append({
                "path": str(path),
                "variable": variable,
                "shape": list(records[row, :, 0].shape),
                "dtype": str(records.dtype),
                "sha256": sha256(path),
            })
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--initial-ids", type=parse_ids, default=parse_ids("50-89"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-kind", choices=["grf", "iid"], default="grf")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="torch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--solver-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--output-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--limit-systems", type=int, default=None,
                        help="development-only cap after applying --split")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = FHNConfig(
        grid_size=args.grid_size,
        dt=args.dt,
        steps=args.steps,
        sample_every=args.sample_every,
    )
    params = tuple(p for values in SPLITS.values() for p in values)
    if args.split != "all":
        params = SPLITS[args.split]
    if args.limit_systems is not None:
        if args.limit_systems < 1:
            raise ValueError("--limit-systems must be positive")
        params = params[: args.limit_systems]

    max_id = max(args.initial_ids)
    if args.initial_kind == "grf":
        bank = gaussian_random_fields(
            max_id, config.grid_size, seed=args.seed, dtype=np.float64
        )
    else:
        bank = iid_gaussian_initial_conditions(
            max_id, config.grid_size, seed=args.seed, dtype=np.float64
        )
    selected = bank[np.asarray(args.initial_ids) - 1]

    manifest: dict[str, object] = {
        "schema": "nod-fhn-python-port-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "backend": args.backend,
        "device": args.device if args.backend == "torch" else "cpu",
        "solver_dtype": args.solver_dtype,
        "output_dtype": args.output_dtype,
        "initial_kind": args.initial_kind,
        "initial_seed": args.seed,
        "initial_ids": args.initial_ids,
        "config": {
            "grid_size": config.grid_size,
            "dt": config.dt,
            "steps": config.steps,
            "sample_every": config.sample_every,
            "dx": config.dx,
            "du": config.diffusion_u,
            "dv": config.diffusion_v,
        },
        "parameters": [list(pair) for pair in params],
        "files": [],
    }
    manifest_path = args.output_dir / "generation_manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, (k, beta) in enumerate(params, 1):
        print(f"[{index}/{len(params)}] k={k:.2f}, beta={beta:.2f}", flush=True)
        if args.backend == "torch":
            records = torch_simulate(
                selected,
                k=k,
                beta=beta,
                config=config,
                device=args.device,
                solver_dtype=args.solver_dtype,
                output_dtype=args.output_dtype,
            )
        else:
            records = simulate(
                selected,
                k=k,
                beta=beta,
                config=config,
                output_dtype=np.dtype(args.output_dtype),
            )
        manifest["files"].extend(
            save_loader_files(
                args.output_dir,
                records,
                k=k,
                beta=beta,
                initial_ids=args.initial_ids,
                compress=args.compress,
            )
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
