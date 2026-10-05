import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from scipy.stats import qmc


def load_spec(path: Path) -> Dict[str, Any]:
    # The canonical YAML intentionally uses the JSON subset of YAML so the
    # bootstrap path has no additional parser dependency.
    return json.loads(path.read_text(encoding="utf-8"))


def _map_unit_cube(unit: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    envelope = spec["development_prior_envelope"]
    v01 = spec["development_v0_1"]
    result = np.empty_like(unit)
    for index, name in enumerate(("m_L", "m_R", "b_L", "b_R")):
        bounds = envelope[name]
        result[:, index] = bounds["low"] + unit[:, index] * (bounds["high"] - bounds["low"])
    for index, bounds in ((4, v01["k_c_range"]), (5, v01["d_c_range"])):
        low, high = np.log(np.asarray(bounds, dtype=float))
        result[:, index] = np.exp(low + unit[:, index] * (high - low))
    return result


def sobol_systems(count: int, seed: int, spec: Dict[str, Any]) -> np.ndarray:
    if count <= 0 or count & (count - 1):
        raise ValueError("Sobol count must be a positive power of two")
    unit = qmc.Sobol(d=6, scramble=True, seed=seed).random_base2(int(np.log2(count)))
    return _map_unit_cube(unit, spec)


def _rows(values: np.ndarray, prefix: str, block: int | None = None) -> List[Dict[str, Any]]:
    names = ("m_L", "m_R", "b_L", "b_R", "k_c", "d_c")
    rows = []
    for index, vector in enumerate(values):
        row: Dict[str, Any] = {"system_id": f"{prefix}_{index:04d}"}
        if block is not None:
            row["sobol_block"] = block
        row.update({name: float(value) for name, value in zip(names, vector)})
        rows.append(row)
    return rows


def write_development_and_particles(spec_path: Path, output_root: Path) -> Dict[str, Any]:
    spec = load_spec(spec_path)
    if spec["access"]["discovery"] or spec["access"]["design_validation"] or spec["access"]["sealed"]:
        raise RuntimeError("development bootstrap requires all reserved splits to remain closed")
    output_root.mkdir(parents=True, exist_ok=True)
    spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    development = sobol_systems(
        spec["splits"]["development"], spec["splits"]["development_seed"], spec
    )
    development_payload = {
        "schema_version": "1.0",
        "split": "development",
        "adaptive_use_authorized": True,
        "spec_sha256": spec_hash,
        "systems": _rows(development, "dev"),
    }
    development_path = output_root / "development_v0_1.json"
    development_path.write_text(json.dumps(development_payload, indent=2, sort_keys=True) + "\n")

    particle_rows: List[Dict[str, Any]] = []
    block_count = len(spec["splits"]["posterior_seed_blocks"])
    per_block = spec["splits"]["posterior_particles"] // block_count
    for block, seed in enumerate(spec["splits"]["posterior_seed_blocks"]):
        values = sobol_systems(per_block, seed, spec)
        block_rows = _rows(values, f"particle_b{block}", block)
        particle_rows.extend(block_rows)
    particle_payload = {
        "schema_version": "1.0",
        "role": "posterior_integration_particles_not_evaluation_systems",
        "spec_sha256": spec_hash,
        "particles": particle_rows,
    }
    particle_path = output_root / "posterior_particles_v0_1.json"
    particle_path.write_text(json.dumps(particle_payload, indent=2, sort_keys=True) + "\n")
    receipt = {
        "status": "DEVELOPMENT_ONLY_MANIFESTS_CREATED",
        "spec_sha256": spec_hash,
        "development_count": len(development),
        "posterior_particle_count": len(particle_rows),
        "discovery_generated": False,
        "design_validation_generated": False,
        "sealed_generated": False,
        "files": {
            development_path.name: hashlib.sha256(development_path.read_bytes()).hexdigest(),
            particle_path.name: hashlib.sha256(particle_path.read_bytes()).hexdigest(),
        },
    }
    (output_root / "manifest_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_development_and_particles(args.spec, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
