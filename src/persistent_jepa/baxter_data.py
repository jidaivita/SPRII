"""Frozen Baxter hardness bank and balanced relation pairing for Paper A A1."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import torch


SHAPES = ("cube", "cylinder")
HARDNESS_LEVELS = (0, 1, 2)
CONDITIONS = ("Random", "R_H", "R_S")
CONFIGS = tuple(f"{shape}_h{level}" for shape in SHAPES for level in HARDNESS_LEVELS)
MEASURED_HARDNESS = {
    "cube_h0": 4.13,
    "cube_h1": 8.52,
    "cube_h2": 12.21,
    "cylinder_h0": 4.62,
    "cylinder_h1": 8.22,
    "cylinder_h2": 12.07,
}
TARGET_INDICES = (40, 59, 79)


@dataclass(frozen=True)
class BaxterRecord:
    record_id: str
    config_id: str
    shape: str
    hardness_level: int
    measured_hardness: float
    grasp_id: int
    split: str
    relative_path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: dict) -> "BaxterRecord":
        return cls(**value)


@dataclass
class BaxterBatch:
    query_history: torch.Tensor
    query_targets: torch.Tensor
    donor_history: torch.Tensor
    query_config: torch.Tensor
    donor_config: torch.Tensor
    hardness_level: torch.Tensor
    shape: torch.Tensor

    def to(self, device: torch.device) -> "BaxterBatch":
        return BaxterBatch(**{key: value.to(device) for key, value in self.__dict__.items()})


def find_dataset_root(extracted_root: Path) -> Path:
    candidates = sorted(path for path in extracted_root.rglob("Dataset") if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(f"expected one Dataset directory under {extracted_root}, found {candidates}")
    return candidates[0]


def parse_grasp_id(path: Path) -> int:
    match = re.search(r"(\d+)\.csv$", path.name)
    if match is None:
        raise ValueError(f"cannot parse grasp id from {path.name}")
    return int(match.group(1))


def load_peak(path: Path) -> np.ndarray:
    with path.open(newline="") as handle:
        value = np.asarray([[float(item) for item in row] for row in csv.reader(handle)], dtype=np.float32)
    if value.shape != (80, 16):
        raise ValueError(f"expected an 80x16 peak, got {value.shape} for {path}")
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite tactile value in {path}")
    return value


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema_version") != "paper-a-a1-baxter-v1.0":
        raise ValueError("unexpected Baxter manifest schema")
    if value.get("source_archive_sha256") != (
        "a7d3782b29df55d46313ca0d3a9b1bd37a400fc15bcb4af12fc6072533f2643c"
    ):
        raise ValueError("Baxter source archive hash is not the frozen value")
    return value


def records_for_split(manifest: dict, split: str) -> list[BaxterRecord]:
    if split not in {"train", "validation", "confirmation"}:
        raise ValueError(f"unknown split {split}")
    records = [BaxterRecord.from_dict(row) for row in manifest["records"] if row["split"] == split]
    expected = {"train": 660, "validation": 180, "confirmation": 180}[split]
    if len(records) != expected:
        raise ValueError(f"split {split} has {len(records)} records, expected {expected}")
    return records


class BaxterSplit:
    """Loads one frozen split and creates exactly balanced 48-pair batches."""

    def __init__(
        self,
        data_root: Path,
        manifest_path: Path,
        normalization_path: Path,
        split: str,
        condition: str,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown Baxter condition {condition}")
        self.data_root = data_root
        self.condition = condition
        self.manifest = load_manifest(manifest_path)
        self.records = records_for_split(self.manifest, split)
        self.by_config: dict[str, list[BaxterRecord]] = {config: [] for config in CONFIGS}
        for record in self.records:
            self.by_config[record.config_id].append(record)
        expected_per_config = {"train": 110, "validation": 30, "confirmation": 30}[split]
        for config, rows in self.by_config.items():
            if len(rows) != expected_per_config:
                raise ValueError(f"{config} has {len(rows)} rows, expected {expected_per_config}")
        normalization = json.loads(normalization_path.read_text())
        self.mean = np.asarray(normalization["channel_mean"], dtype=np.float32)
        self.std = np.asarray(normalization["channel_std"], dtype=np.float32)
        if self.mean.shape != (16,) or self.std.shape != (16,) or np.any(self.std <= 0):
            raise ValueError("invalid Baxter train-only normalization")

    def _load(self, record: BaxterRecord) -> np.ndarray:
        value = load_peak(self.data_root / record.relative_path)
        return (value - self.mean) / self.std

    @staticmethod
    def _shape_and_level(config: str) -> tuple[str, int]:
        shape, level = config.rsplit("_h", 1)
        return shape, int(level)

    def _donor_config_blocks(self, rng: np.random.Generator, repeats: int) -> list[dict[str, str]]:
        blocks: list[dict[str, str]] = []
        if self.condition == "R_H":
            for _ in range(repeats):
                blocks.append({
                    config: f"{'cylinder' if config.startswith('cube') else 'cube'}_h{config[-1]}"
                    for config in CONFIGS
                })
            return blocks
        if self.condition == "R_S":
            offsets = np.asarray(([1] * (repeats // 2)) + ([2] * (repeats - repeats // 2)))
            rng.shuffle(offsets)
            for offset in offsets:
                blocks.append({
                    config: f"{self._shape_and_level(config)[0]}_h{(self._shape_and_level(config)[1] + int(offset)) % 3}"
                    for config in CONFIGS
                })
            return blocks
        config_array = np.asarray(CONFIGS)
        for _ in range(repeats):
            while True:
                donor = rng.permutation(config_array)
                if np.all(donor != config_array):
                    blocks.append(dict(zip(CONFIGS, donor.tolist(), strict=True)))
                    break
        return blocks

    def batch(self, batch_pairs: int, batch_seed: int) -> BaxterBatch:
        if batch_pairs % len(CONFIGS) != 0:
            raise ValueError("batch_pairs must be divisible by six configurations")
        repeats = batch_pairs // len(CONFIGS)
        rng = np.random.default_rng(batch_seed)
        blocks = self._donor_config_blocks(rng, repeats)
        chosen_query: dict[str, list[BaxterRecord]] = {}
        for config in CONFIGS:
            rows = self.by_config[config]
            indices = rng.choice(len(rows), size=repeats, replace=repeats > len(rows))
            chosen_query[config] = [rows[int(index)] for index in indices]
        queries: list[BaxterRecord] = []
        donors: list[BaxterRecord] = []
        for block_index, mapping in enumerate(blocks):
            for config in CONFIGS:
                query = chosen_query[config][block_index]
                donor_pool = self.by_config[mapping[config]]
                donor = donor_pool[int(rng.integers(0, len(donor_pool)))]
                queries.append(query)
                donors.append(donor)
        order = rng.permutation(batch_pairs)
        queries = [queries[int(index)] for index in order]
        donors = [donors[int(index)] for index in order]
        query_value = np.stack([self._load(row) for row in queries])
        donor_value = np.stack([self._load(row) for row in donors])
        config_index = {config: index for index, config in enumerate(CONFIGS)}
        return BaxterBatch(
            query_history=torch.from_numpy(query_value[:, :40]),
            query_targets=torch.from_numpy(query_value[:, TARGET_INDICES]),
            donor_history=torch.from_numpy(donor_value[:, :40]),
            query_config=torch.tensor([config_index[row.config_id] for row in queries]),
            donor_config=torch.tensor([config_index[row.config_id] for row in donors]),
            hardness_level=torch.tensor([row.hardness_level for row in queries]),
            shape=torch.tensor([SHAPES.index(row.shape) for row in queries]),
        )

    def classification_batch(
        self, batch_size: int, batch_seed: int, input_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size % len(CONFIGS) != 0:
            raise ValueError("classification batch must be divisible by six configurations")
        if input_length not in {8, 80}:
            raise ValueError("registered certificate lengths are 8 and 80")
        per_config = batch_size // len(CONFIGS)
        rng = np.random.default_rng(batch_seed)
        chosen: list[BaxterRecord] = []
        for config in CONFIGS:
            rows = self.by_config[config]
            indices = rng.choice(len(rows), size=per_config, replace=per_config > len(rows))
            chosen.extend(rows[int(index)] for index in indices)
        order = rng.permutation(len(chosen))
        chosen = [chosen[int(index)] for index in order]
        value = np.stack([self._load(row)[:input_length] for row in chosen])
        return (
            torch.from_numpy(value),
            torch.tensor([row.hardness_level for row in chosen]),
            torch.tensor([SHAPES.index(row.shape) for row in chosen]),
        )

    def iter_records(self) -> Iterable[tuple[BaxterRecord, np.ndarray]]:
        for record in sorted(self.records, key=lambda row: row.record_id):
            yield record, self._load(record)
