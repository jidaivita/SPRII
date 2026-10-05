#!/usr/bin/env python3
"""Generate the frozen PokeWorld unseen-physics datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persistent_jepa.pokeworld import PokeConfig, generate_pokeworld_ood, save_pokeworld


PROTOCOLS = {
    "interpolation": {
        "seed": 20260826,
        "support": ((0.5, 1.5), (2.5, 4.0)),
        "ood": ((1.5, 2.5),),
        "support_component_weights": (0.4, 0.6),
    },
    "extrapolation": {
        "seed": 20260827,
        "support": ((0.5, 3.0),),
        "ood": ((3.0, 4.0),),
        "support_component_weights": (1.0,),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset root: {args.output}")
    protocol = PROTOCOLS[args.protocol]
    cfg = PokeConfig(seed=protocol["seed"])
    dataset = generate_pokeworld_ood(cfg, protocol["support"], protocol["ood"])
    metadata = {
        "schema_version": "pokeworld-ood-1.0",
        "protocol": args.protocol,
        "support_gamma_intervals": protocol["support"],
        "ood_gamma_intervals": protocol["ood"],
        "support_component_weights": protocol["support_component_weights"],
        "sampling": "uniform over union, component probability proportional to interval width",
        "ood_split_role": "sealed evaluation only; never checkpoint or hyperparameter selection",
    }
    manifest = save_pokeworld(args.output, dataset, cfg, metadata=metadata)
    summary = {
        "root": str(args.output),
        "manifest_sha256": manifest["manifest_sha256"],
        "gamma": {
            split: {
                "min": float(payload["gamma"].min()),
                "max": float(payload["gamma"].max()),
                "mean": float(payload["gamma"].mean()),
                "systems": int(payload["gamma"].size),
            }
            for split, payload in dataset.items()
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
