"""Select one NOD and one SPRII checkpoint using only held-out ID initial 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path: Path) -> dict:
    result = json.loads(path.read_text())
    components = {}
    for row in result["aggregate"]:
        if row["split"] == "train" and row["metric"] == "l2_relative":
            components[str(row["horizon"])] = float(row["mean"])
    if set(components) != {"1", "5", "50"}:
        raise ValueError(f"incomplete validation result: {path}")
    return {
        "result": str(path),
        "checkpoint": result["checkpoint"],
        "step": result["checkpoint_step"],
        "l2_relative": components,
        "selection_score": float(np.mean(list(components.values()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [summarize(path) for path in sorted(args.validation_dir.glob("*_val_id5.json"))]
    if len(rows) != 16:
        raise ValueError(f"expected 16 validation files, found {len(rows)}")
    nod = [row for row in rows if Path(row["result"]).name.startswith("nod_")]
    sprii = [row for row in rows if not Path(row["result"]).name.startswith("nod_")]
    selection = {
        "selection_data": "ID systems, target initial_id=5, conditioning initial_id=36",
        "selection_metric": "unweighted mean L2 relative error across H=1,5,50",
        "nod": min(nod, key=lambda row: row["selection_score"]),
        "sprii": min(sprii, key=lambda row: row["selection_score"]),
        "all_candidates": sorted(rows, key=lambda row: row["selection_score"]),
    }
    args.output.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
