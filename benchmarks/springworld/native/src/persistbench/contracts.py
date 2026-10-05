"""Versioned, model-agnostic public contracts for PersistBench."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    CHALLENGE = "challenge"


class TaskType(str, Enum):
    PREDICTION = "prediction"
    REPRESENTATION = "representation"
    CONTROL = "control"


# Backward-compatible name during the v0.1 schema migration.
TaskMode = TaskType


class InteractionMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"


class DataModality(str, Enum):
    SIMULATED = "simulated"
    REAL = "real"
    HYBRID = "hybrid"


class OutputType(str, Enum):
    PREDICTION = "prediction"
    REPRESENTATION = "representation"
    ACTION = "action"


class ComputeTier(str, Enum):
    LITE = "lite"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True)
class ComputeBudgetSpec:
    tier: ComputeTier
    max_wall_seconds: int
    max_cpu_cores: int
    max_gpus: int
    max_memory_gb: int
    training_included: bool
    reference_hardware: str
    timeout_policy: str = "fail_run"

    def validate(self) -> None:
        if self.max_wall_seconds <= 0 or self.max_cpu_cores <= 0 or self.max_memory_gb <= 0:
            raise ValueError("compute budget limits must be positive")
        if self.max_gpus < 0 or not self.reference_hardware:
            raise ValueError("invalid compute hardware budget")
        if self.timeout_policy != "fail_run":
            raise ValueError("v0.2 compute timeout policy must be fail_run")


class Capability(str, Enum):
    RECOVERABILITY = "recoverability"
    FORMATION = "formation"
    ORGANIZATION = "organization"
    DELAYED_VALUE = "delayed_value"
    LEARNER_UTILITY = "learner_utility"
    SYSTEM_SPECIFICITY = "system_specificity"
    OOD_TRANSFER = "ood_transfer"


class AdmissionState(str, Enum):
    CORE_CANDIDATE = "core_candidate"
    CHALLENGE_CANDIDATE = "challenge_candidate"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    REAL_DATA_EXTENSION = "real_data_extension"
    PARKED_PENDING_AUDIT = "parked_pending_audit"


class ReleaseState(str, Enum):
    INCUBATING = "incubating"
    READY = "ready"
    BLOCKED = "blocked"


class FailureLayer(str, Enum):
    NONE = "none"
    ENVIRONMENT_VALIDITY = "environment_validity"
    ASSAY_IDENTIFIABILITY = "assay_identifiability"
    POLICY_COMPETENCY = "policy_competency"
    IMPLEMENTATION = "implementation"
    COMPUTE_FEASIBILITY = "compute_feasibility"
    SCIENTIFIC_OUTCOME = "scientific_outcome"
    DATA_AVAILABILITY = "data_availability"


class ControlKind(str, Enum):
    BASELINE = "baseline"
    COUNTERFACTUAL = "counterfactual"
    CONDITION = "condition"
    QUALIFICATION = "qualification"


class MetricDirection(str, Enum):
    LOWER = "lower"
    HIGHER = "higher"
    DESCRIPTIVE = "descriptive"


class MetricScope(str, Enum):
    CASE = "case"
    SUITE = "suite"


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    display_name: str
    source_papers: tuple[str, ...]
    asset_kind: str
    data_modality: DataModality
    source_ref: str
    admission: AdmissionState
    status_note: str = ""
    fresh_test_required: bool = True

    def validate(self) -> None:
        if not self.environment_id or "/" in self.environment_id:
            raise ValueError("environment_id must be a non-empty slug")
        if not self.source_papers:
            raise ValueError("at least one source paper is required")
        if not self.source_ref:
            raise ValueError("source_ref is required")


@dataclass(frozen=True)
class ControlSpec:
    control_id: str
    kind: ControlKind

    def validate(self) -> None:
        if not self.control_id:
            raise ValueError("control_id is required")


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    implementation: str
    direction: MetricDirection
    unit: str
    scope: MetricScope = MetricScope.CASE
    version: str = "1.0"
    per_system_aggregation: str = "mean"
    suite_aggregation: str = "macro"
    uncertainty: str = "system_bootstrap"
    primary: bool = True
    output_key: str | None = None
    non_finite_policy: str = "fail_run"
    missing_policy: str = "fail_run"
    minimum_groups: int = 1

    def validate(self) -> None:
        if not self.metric_id or not self.implementation or not self.version:
            raise ValueError("metric id, implementation, and version are required")
        if not self.unit:
            raise ValueError("metric unit is required")
        if self.non_finite_policy != "fail_run" or self.missing_policy != "fail_run":
            raise ValueError("v0.2 only supports explicit fail_run metric policies")
        if self.minimum_groups <= 0:
            raise ValueError("minimum_groups must be positive")


@dataclass(frozen=True)
class OutputSpec:
    output_type: OutputType
    required_keys: tuple[str, ...]
    dtype: str
    representation_dim: int | None = None
    shape: tuple[int, ...] | None = None

    def validate(self) -> None:
        if not self.required_keys or self.dtype not in {"float32", "float64"}:
            raise ValueError("output keys and supported floating dtype are required")
        if self.output_type is OutputType.REPRESENTATION and (
            self.representation_dim is None or self.representation_dim <= 0
        ):
            raise ValueError("representation outputs require a positive fixed dimension")
        if self.shape is not None and (len(self.required_keys) != 1 or any(value <= 0 for value in self.shape)):
            raise ValueError("a fixed output shape requires one key and positive dimensions")


@dataclass(frozen=True)
class AssaySpec:
    assay_id: str
    environment_id: str
    protocol_version: str
    task_type: TaskType
    interaction_mode: InteractionMode
    capabilities: tuple[Capability, ...]
    controls: tuple[ControlSpec, ...]
    metrics: tuple[MetricSpec, ...]
    group_split_keys: tuple[str, ...]
    output_spec: OutputSpec | None = None
    release_state: ReleaseState = ReleaseState.INCUBATING
    leaderboard: bool = False
    compute_tiers: tuple[ComputeTier, ...] = (ComputeTier.LITE,)

    @property
    def primary_metrics(self) -> tuple[MetricSpec, ...]:
        return tuple(metric for metric in self.metrics if metric.primary)

    def validate(self) -> None:
        if not self.assay_id.startswith(f"{self.environment_id}/"):
            raise ValueError("assay_id must be namespaced by environment_id")
        if not self.protocol_version:
            raise ValueError("protocol_version is required")
        if not self.capabilities:
            raise ValueError("an assay must measure at least one capability")
        if not self.controls:
            raise ValueError("an assay must declare at least one typed control")
        if not self.primary_metrics:
            raise ValueError("an assay must declare at least one primary metric")
        if not self.group_split_keys:
            raise ValueError("group_split_keys are required")
        if not self.compute_tiers:
            raise ValueError("an assay must expose at least one compute tier")
        if self.leaderboard and self.release_state is not ReleaseState.READY:
            raise ValueError("only release-ready assays may enter the leaderboard")
        for control in self.controls:
            control.validate()
        for metric in self.metrics:
            metric.validate()
        if self.output_spec is not None:
            self.output_spec.validate()
            expected = {
                TaskType.PREDICTION: OutputType.PREDICTION,
                TaskType.REPRESENTATION: OutputType.REPRESENTATION,
                TaskType.CONTROL: OutputType.ACTION,
            }[self.task_type]
            if self.output_spec.output_type is not expected:
                raise ValueError("output spec type is incompatible with task type")


@dataclass(frozen=True)
class FailureEvidence:
    assay_id: str
    protocol_version: str
    layer: FailureLayer
    evidence_ref: str
    note: str


@dataclass(frozen=True)
class DerivedMetricSpec:
    metric_id: str
    base_metric_id: str
    left_condition: str
    right_condition: str
    operation: str = "left_minus_right"
    bootstrap_samples: int = 2000

    def validate(self) -> None:
        if not all(
            (self.metric_id, self.base_metric_id, self.left_condition, self.right_condition)
        ):
            raise ValueError("derived metric identifiers and conditions are required")
        if self.operation != "left_minus_right" or self.bootstrap_samples <= 0:
            raise ValueError("unsupported derived metric operation or bootstrap budget")


@dataclass(frozen=True)
class ExperienceBatch:
    system_keys: Sequence[str]
    interaction_keys: Sequence[str]
    observations: Any
    actions: Any
    masks: Any | None = None
    timestamps: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.system_keys or not self.interaction_keys:
            raise ValueError("experience identities cannot be empty")
        allowed = {"observation_schema", "action_schema", "padding", "public_tags"}
        unexpected = set(self.metadata).difference(allowed)
        if unexpected:
            raise ValueError(f"model-facing metadata is not allowlisted: {sorted(unexpected)}")


@dataclass(frozen=True)
class QueryBatch:
    query_keys: Sequence[str]
    episode_tokens: Sequence[str]
    observations: Any
    actions: Any | None = None
    horizons: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.query_keys or not self.episode_tokens:
            raise ValueError("query identities and opaque episode tokens cannot be empty")
        allowed = {"observation_schema", "action_schema", "public_tags"}
        unexpected = set(self.metadata).difference(allowed)
        if unexpected:
            raise ValueError(f"model-facing query metadata is not allowlisted: {sorted(unexpected)}")


@dataclass(frozen=True)
class AgentOutput:
    output_type: OutputType
    values: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunContext:
    assay_id: str
    protocol_version: str
    split: Split
    compute_tier: ComputeTier
    seed: int


@dataclass(frozen=True)
class EpisodeContext:
    episode_token: str


@dataclass(frozen=True)
class CapabilityDeclaration:
    output_types: tuple[OutputType, ...]
    representation_dim: int | None = None
    output_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationRequest:
    assay_id: str
    split: Split
    compute_tier: ComputeTier
    seed: int
    limit: int | None = None
    test_authorization: str | None = field(default=None, repr=False)

    def validate(self) -> None:
        if not self.assay_id or "/" not in self.assay_id:
            raise ValueError("assay_id must be namespaced")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.split in {Split.TEST, Split.CHALLENGE} and not self.test_authorization:
            raise PermissionError("sealed splits require controlled-runner authorization")


@dataclass(frozen=True)
class EvaluationCase:
    """Evaluator-private case; only experience/query cross the agent boundary."""

    case_id: str
    group_key: str
    episode_token: str
    condition: str
    experience: ExperienceBatch
    query: QueryBatch
    targets: Mapping[str, Any]
    private_metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.case_id or not self.group_key or not self.episode_token or not self.condition:
            raise ValueError("case, group, episode, and condition identities are required")
        if not self.targets:
            raise ValueError("evaluator-owned targets are required")
        if set(self.query.episode_tokens) != {self.episode_token}:
            raise ValueError("query episode token disagrees with evaluator case")
        self.experience.validate()
        self.query.validate()


@dataclass(frozen=True)
class MetricRecord:
    assay_id: str
    case_id: str
    condition: str
    metric: str
    metric_version: str
    value: float
    group_key: str


@runtime_checkable
class AssayAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def iter_cases(self, request: EvaluationRequest) -> Iterable[EvaluationCase]:
        ...

    def provenance(self) -> Mapping[str, Any]:
        ...


@runtime_checkable
class PersistentAgent(Protocol):
    """Shared lifecycle; task-specific output is declared and type checked."""

    def initialize(self, context: RunContext) -> CapabilityDeclaration:
        ...

    def reset(self, context: EpisodeContext) -> None:
        ...

    def ingest(self, experience: ExperienceBatch) -> None:
        ...

    def respond(self, query: QueryBatch) -> AgentOutput:
        ...
