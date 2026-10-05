"""Load and validate the portable benchmark registry."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    AdmissionState,
    AssaySpec,
    Capability,
    ComputeTier,
    ComputeBudgetSpec,
    ControlKind,
    ControlSpec,
    DataModality,
    EnvironmentSpec,
    FailureEvidence,
    FailureLayer,
    InteractionMode,
    MetricDirection,
    MetricScope,
    MetricSpec,
    OutputSpec,
    OutputType,
    ReleaseState,
    TaskType,
)


REGISTRY_SCHEMA = "persistbench.registry.v0.2"


@dataclass(frozen=True)
class SourceSpec:
    source_ref: str
    source_kind: str
    immutable_ref: str
    redistribution: str
    note: str = ""

    def validate(self) -> None:
        if not self.source_ref or not self.source_kind or not self.immutable_ref:
            raise ValueError("source_ref, source_kind, and immutable_ref are required")
        if self.redistribution not in {"original_only", "allowed", "downloader_only", "unresolved"}:
            raise ValueError(f"invalid redistribution state: {self.redistribution}")


@dataclass(frozen=True)
class BenchmarkRegistry:
    schema: str
    environments: tuple[EnvironmentSpec, ...]
    assays: tuple[AssaySpec, ...]
    sources: tuple[SourceSpec, ...]
    failures: tuple[FailureEvidence, ...]
    compute_budgets: tuple[ComputeBudgetSpec, ...]
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkRegistry":
        source_path = Path(path).resolve()
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if payload.get("schema") != REGISTRY_SCHEMA:
            raise ValueError(f"expected registry schema {REGISTRY_SCHEMA}")
        metric_catalog = {
            name: _metric(name, item) for name, item in payload["metric_catalog"].items()
        }
        registry = cls(
            schema=payload["schema"],
            environments=tuple(_environment(item) for item in payload["environments"]),
            assays=tuple(_assay(item, metric_catalog) for item in payload["assays"]),
            sources=tuple(SourceSpec(**item) for item in payload["sources"]),
            failures=tuple(_failure(item) for item in payload.get("failure_evidence", [])),
            compute_budgets=tuple(_compute_budget(item) for item in payload["compute_tiers"]),
            source_path=source_path,
        )
        registry.validate()
        return registry

    def validate(self) -> None:
        environment_ids = [item.environment_id for item in self.environments]
        assay_ids = [item.assay_id for item in self.assays]
        source_ids = [item.source_ref for item in self.sources]
        compute_tiers = [item.tier for item in self.compute_budgets]
        for values, label in (
            (environment_ids, "environment_id"),
            (assay_ids, "assay_id"),
            (source_ids, "source_ref"),
            (compute_tiers, "compute_tier"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label}")
        known_sources = set(source_ids)
        known_compute_tiers = set(compute_tiers)
        for budget in self.compute_budgets:
            budget.validate()
        for source in self.sources:
            source.validate()
        for environment in self.environments:
            environment.validate()
            if environment.source_ref not in known_sources:
                raise ValueError(f"unknown source_ref: {environment.source_ref}")
        known_environments = set(environment_ids)
        known_assays = set(assay_ids)
        for assay in self.assays:
            assay.validate()
            if assay.environment_id not in known_environments:
                raise ValueError(f"unknown assay environment: {assay.environment_id}")
            if not set(assay.compute_tiers).issubset(known_compute_tiers):
                raise ValueError(f"assay references an undefined compute tier: {assay.assay_id}")
        for failure in self.failures:
            if failure.assay_id not in known_assays:
                raise ValueError(f"failure evidence references unknown assay: {failure.assay_id}")

    def summary(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "environment_count": len(self.environments),
            "assay_count": len(self.assays),
            "leaderboard_assay_count": sum(item.leaderboard for item in self.assays),
            "release_ready_assay_count": sum(
                item.release_state is ReleaseState.READY for item in self.assays
            ),
            "admission_counts": {
                state.value: sum(item.admission is state for item in self.environments)
                for state in AdmissionState
            },
            "release_counts": {
                state.value: sum(item.release_state is state for item in self.assays)
                for state in ReleaseState
            },
            "failure_evidence_counts": {
                layer.value: sum(item.layer is layer for item in self.failures)
                for layer in FailureLayer
            },
            "compute_tiers": [item.tier.value for item in self.compute_budgets],
        }


def _environment(item: dict[str, Any]) -> EnvironmentSpec:
    return EnvironmentSpec(
        environment_id=item["environment_id"],
        display_name=item["display_name"],
        source_papers=tuple(item["source_papers"]),
        asset_kind=item["asset_kind"],
        data_modality=DataModality(item["data_modality"]),
        source_ref=item["source_ref"],
        admission=AdmissionState(item["admission"]),
        status_note=item.get("status_note", ""),
        fresh_test_required=bool(item.get("fresh_test_required", True)),
    )


def _metric(metric_id: str, item: dict[str, Any]) -> MetricSpec:
    return MetricSpec(
        metric_id=metric_id,
        implementation=item["implementation"],
        direction=MetricDirection(item["direction"]),
        unit=item["unit"],
        scope=MetricScope(item.get("scope", "case")),
        version=item.get("version", "1.0"),
        per_system_aggregation=item.get("per_system_aggregation", "mean"),
        suite_aggregation=item.get("suite_aggregation", "macro"),
        uncertainty=item.get("uncertainty", "system_bootstrap"),
        primary=bool(item.get("primary", True)),
        output_key=item.get("output_key"),
        non_finite_policy=item.get("non_finite_policy", "fail_run"),
        missing_policy=item.get("missing_policy", "fail_run"),
        minimum_groups=int(item.get("minimum_groups", 1)),
    )


def _assay(item: dict[str, Any], catalog: dict[str, MetricSpec]) -> AssaySpec:
    try:
        metrics = tuple(catalog[name] for name in item["metrics"])
    except KeyError as error:
        raise ValueError(f"unknown metric reference: {error.args[0]}") from error
    return AssaySpec(
        assay_id=item["assay_id"],
        environment_id=item["environment_id"],
        protocol_version=item["protocol_version"],
        task_type=TaskType(item["task_type"]),
        interaction_mode=InteractionMode(item["interaction_mode"]),
        capabilities=tuple(Capability(value) for value in item["capabilities"]),
        controls=tuple(
            ControlSpec(control_id=control["control_id"], kind=ControlKind(control["kind"]))
            for control in item["controls"]
        ),
        metrics=metrics,
        group_split_keys=tuple(item["group_split_keys"]),
        output_spec=_output_spec(item["output_spec"]) if "output_spec" in item else None,
        release_state=ReleaseState(item.get("release_state", "incubating")),
        leaderboard=bool(item.get("leaderboard", False)),
        compute_tiers=tuple(ComputeTier(value) for value in item.get("compute_tiers", ["lite"])),
    )


def _output_spec(item: dict[str, Any]) -> OutputSpec:
    return OutputSpec(
        output_type=OutputType(item["output_type"]),
        required_keys=tuple(item["required_keys"]),
        dtype=item["dtype"],
        representation_dim=item.get("representation_dim"),
        shape=tuple(item["shape"]) if "shape" in item else None,
    )


def _failure(item: dict[str, Any]) -> FailureEvidence:
    return FailureEvidence(
        assay_id=item["assay_id"],
        protocol_version=item["protocol_version"],
        layer=FailureLayer(item["layer"]),
        evidence_ref=item["evidence_ref"],
        note=item["note"],
    )


def _compute_budget(item: dict[str, Any]) -> ComputeBudgetSpec:
    return ComputeBudgetSpec(
        tier=ComputeTier(item["tier"]),
        max_wall_seconds=int(item["max_wall_seconds"]),
        max_cpu_cores=int(item["max_cpu_cores"]),
        max_gpus=int(item["max_gpus"]),
        max_memory_gb=int(item["max_memory_gb"]),
        training_included=bool(item["training_included"]),
        reference_hardware=item["reference_hardware"],
        timeout_policy=item.get("timeout_policy", "fail_run"),
    )


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("usage: python -m persistbench.registry REGISTRY.json")
    registry = BenchmarkRegistry.load(arguments[0])
    print(json.dumps(registry.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
