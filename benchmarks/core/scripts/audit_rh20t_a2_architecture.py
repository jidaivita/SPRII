#!/usr/bin/env python3
"""Architecture-only A2 fairness audit; reads no data or checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persistent_jepa.rh20t_model import RH20TJEPA, RH20TModelConfig


def count(model: RH20TJEPA) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    b3 = RH20TJEPA(RH20TModelConfig("B3-Indep"))
    mono = RH20TJEPA(RH20TModelConfig("Mono-QD-Indep"))
    b3_count, mono_count = count(b3), count(mono)
    relative = abs(mono_count - b3_count) / b3_count
    predictor_shapes_equal = all(
        getattr(b3.predictor, name).weight.shape == getattr(mono.predictor, name).weight.shape
        for name in ("latent", "force", "tcp_xyz")
    )
    payload = {
        "schema_version": "1.1",
        "audit_mode": "ARCHITECTURE_ONLY_NO_DATA_NO_CHECKPOINT",
        "b3_trainable_parameters": b3_count,
        "monolithic_qd_trainable_parameters": mono_count,
        "absolute_difference": mono_count - b3_count,
        "relative_absolute_difference": relative,
        "pre_frozen_relative_tolerance": args.relative_tolerance,
        "within_tolerance": relative <= args.relative_tolerance,
        "prediction_head_shapes_equal": predictor_shapes_equal,
        "pass": relative <= args.relative_tolerance and predictor_shapes_equal,
        "model_output_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not payload["pass"]:
        raise SystemExit("A2 architecture fairness audit failed")


if __name__ == "__main__":
    main()
