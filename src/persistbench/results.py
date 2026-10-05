"""Stable, machine-readable run manifests and result bundles."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__

from .contracts import EvaluationRequest, MetricRecord


@dataclass(frozen=True)
class RunManifest:
    schema: str
    run_id: str
    created_at_utc: str
    assay_id: str
    split: str
    compute_tier: str
    seed: int
    agent_id: str
    adapter_id: str
    adapter_version: str
    benchmark_version: str
    protocol_version: str
    evaluator_digest: str
    agent_artifact_digest: str
    agent_checkpoint_digest: str | None
    source_provenance: Mapping[str, Any]
    python_version: str
    platform: str

    @classmethod
    def create(
        cls,
        request: EvaluationRequest,
        *,
        agent_id: str,
        adapter_id: str,
        adapter_version: str,
        source_provenance: Mapping[str, Any],
        protocol_version: str = "unknown",
        evaluator_digest: str = "development",
        agent_artifact_digest: str = "development",
        agent_checkpoint_digest: str | None = None,
    ) -> "RunManifest":
        identity = {
            "assay_id": request.assay_id,
            "split": request.split.value,
            "compute_tier": request.compute_tier.value,
            "seed": request.seed,
            "limit": request.limit,
            "benchmark_version": __version__,
            "agent_id": agent_id,
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "protocol_version": protocol_version,
            "evaluator_digest": evaluator_digest,
            "agent_artifact_digest": agent_artifact_digest,
            "agent_checkpoint_digest": agent_checkpoint_digest,
            "source_provenance": source_provenance,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return cls(
            schema="persistbench.run.v0.2",
            run_id=digest,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            assay_id=request.assay_id,
            split=request.split.value,
            compute_tier=request.compute_tier.value,
            seed=request.seed,
            agent_id=agent_id,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            benchmark_version=__version__,
            protocol_version=protocol_version,
            evaluator_digest=evaluator_digest,
            agent_artifact_digest=agent_artifact_digest,
            agent_checkpoint_digest=agent_checkpoint_digest,
            source_provenance=dict(source_provenance),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
        )


@dataclass(frozen=True)
class ResultBundle:
    manifest: RunManifest
    records: Sequence[MetricRecord]
    aggregates: Mapping[str, float]
    case_commitments: Mapping[str, str]
    # Evaluator-private suite inputs used for paired system bootstrap. They are
    # intentionally omitted from both serialized result views.
    suite_evidence: Mapping[str, Mapping[str, Sequence[Any]]] = field(
        default_factory=dict, repr=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": asdict(self.manifest),
            "records": [asdict(record) for record in self.records],
            "aggregates": dict(self.aggregates),
            "case_commitments": dict(self.case_commitments),
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Leaderboard-safe view without case-level feedback."""
        return {
            "manifest": asdict(self.manifest),
            "aggregates": dict(self.aggregates),
            "evaluated_record_count": len(self.records),
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_public(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_public_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
