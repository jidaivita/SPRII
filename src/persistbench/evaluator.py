"""Evaluator-owned execution, target handling, metrics, and aggregation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import inspect
import json
from typing import Mapping

import numpy as np

from .contracts import (
    AssayAdapter,
    AssaySpec,
    EpisodeContext,
    EvaluationRequest,
    MetricScope,
    MetricRecord,
    OutputType,
    PersistentAgent,
    RunContext,
    TaskType,
)
from .metrics import MetricRegistry
from .results import ResultBundle, RunManifest


class Evaluator:
    def __init__(self, metrics: MetricRegistry) -> None:
        self.metrics = metrics

    def run(
        self,
        *,
        assay: AssaySpec,
        request: EvaluationRequest,
        adapter: AssayAdapter,
        agent: PersistentAgent,
        agent_id: str,
    ) -> ResultBundle:
        request.validate()
        assay.validate()
        if request.assay_id != assay.assay_id:
            raise ValueError("request and assay identifiers disagree")
        if request.compute_tier not in assay.compute_tiers:
            raise ValueError("requested compute tier is unavailable for this assay")
        expected_output = {
            TaskType.PREDICTION: OutputType.PREDICTION,
            TaskType.REPRESENTATION: OutputType.REPRESENTATION,
            TaskType.CONTROL: OutputType.ACTION,
        }[assay.task_type]
        declaration = agent.initialize(
            RunContext(
                assay_id=assay.assay_id,
                protocol_version=assay.protocol_version,
                split=request.split,
                compute_tier=request.compute_tier,
                seed=request.seed,
            )
        )
        if expected_output not in declaration.output_types:
            raise ValueError(f"agent does not declare {expected_output.value} output support")
        if assay.output_spec is not None:
            if (
                assay.output_spec.representation_dim is not None
                and declaration.representation_dim != assay.output_spec.representation_dim
            ):
                raise ValueError("agent declaration has the wrong representation dimension")
            if not set(assay.output_spec.required_keys).issubset(declaration.output_keys):
                raise ValueError("agent declaration omits required output keys")
        records: list[MetricRecord] = []
        case_commitments: dict[str, str] = {}
        seen_cases: set[str] = set()
        suite_pending: dict[tuple[str, str], dict[str, list[object]]] = defaultdict(
            lambda: {"predictions": [], "targets": [], "groups": []}
        )
        for index, case in enumerate(adapter.iter_cases(request)):
            if request.limit is not None and index >= request.limit:
                break
            case.validate()
            if case.case_id in seen_cases:
                raise ValueError(f"duplicate case_id: {case.case_id}")
            seen_cases.add(case.case_id)
            case_commitments[case.case_id] = _case_commitment(case)
            agent.reset(EpisodeContext(episode_token=case.episode_token))
            agent.ingest(case.experience)
            output = agent.respond(case.query)
            if output.output_type is not expected_output:
                raise ValueError("agent returned an output type incompatible with the assay")
            if assay.output_spec is not None:
                if not isinstance(output.values, Mapping):
                    raise TypeError("typed output spec requires a keyed mapping")
                missing_keys = set(assay.output_spec.required_keys).difference(output.values)
                if missing_keys:
                    raise ValueError(f"agent output omits required keys: {sorted(missing_keys)}")
                for key in assay.output_spec.required_keys:
                    values = np.asarray(output.values[key])
                    if str(values.dtype) != assay.output_spec.dtype or not np.all(np.isfinite(values)):
                        raise ValueError("agent output must contain finite floating values")
                    if (
                        assay.output_spec.representation_dim is not None
                        and values.shape != (assay.output_spec.representation_dim,)
                    ):
                        raise ValueError("agent representation has the wrong shape")
                    if assay.output_spec.shape is not None and values.shape != assay.output_spec.shape:
                        raise ValueError("agent output has the wrong fixed shape")
            for metric in assay.primary_metrics:
                target = case.targets[metric.metric_id]
                if metric.output_key is None:
                    if len(assay.primary_metrics) != 1:
                        raise ValueError("multi-metric assays require output_key bindings")
                    prediction = output.values
                else:
                    if not isinstance(output.values, Mapping):
                        raise TypeError("keyed metric output requires a mapping")
                    prediction = output.values[metric.output_key]
                if metric.scope is MetricScope.SUITE:
                    pending = suite_pending[(case.condition, metric.metric_id)]
                    pending["predictions"].append(prediction)
                    pending["targets"].append(target)
                    pending["groups"].append(case.group_key)
                else:
                    value = self.metrics.evaluate(metric.implementation, prediction, target)
                    records.append(
                        MetricRecord(
                            assay_id=assay.assay_id,
                            case_id=case.case_id,
                            condition=case.condition,
                            metric=metric.metric_id,
                            metric_version=metric.version,
                            value=value,
                            group_key=case.group_key,
                        )
                    )
        metric_by_id = {metric.metric_id: metric for metric in assay.primary_metrics}
        for (condition, metric_id), pending in sorted(suite_pending.items()):
            metric = metric_by_id[metric_id]
            unique_groups = set(pending["groups"])
            if len(unique_groups) < metric.minimum_groups:
                raise ValueError(
                    f"metric {metric_id} requires at least {metric.minimum_groups} groups"
                )
            value = self.metrics.evaluate(
                metric.implementation, pending["predictions"], pending["targets"]
            )
            records.append(
                MetricRecord(
                    assay_id=assay.assay_id,
                    case_id="__suite__",
                    condition=condition,
                    metric=metric.metric_id,
                    metric_version=metric.version,
                    value=value,
                    group_key="__suite__",
                )
            )
        if not records:
            raise ValueError("adapter produced no evaluated cases")
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for record in records:
            grouped[(record.condition, record.metric, record.group_key)].append(record.value)
        per_group = {
            key: sum(values) / len(values) for key, values in grouped.items()
        }
        macro_groups: dict[str, list[float]] = defaultdict(list)
        for (condition, metric, _group_key), value in per_group.items():
            macro_groups[f"{condition}/{metric}"].append(value)
        aggregates = {
            name: sum(values) / len(values)
            for name, values in sorted(macro_groups.items())
        }
        manifest = RunManifest.create(
            request,
            agent_id=agent_id,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
            source_provenance=adapter.provenance(),
            protocol_version=assay.protocol_version,
            evaluator_digest=_implementation_digest(
                Evaluator, inspect.getmodule(type(self.metrics)), self.metrics.names
            ),
            agent_artifact_digest=_implementation_digest(type(agent), _public_state(agent)),
            agent_checkpoint_digest=getattr(agent, "checkpoint_digest", None),
        )
        return ResultBundle(
            manifest=manifest,
            records=records,
            aggregates=aggregates,
            case_commitments=case_commitments,
            suite_evidence={
                metric_id: {
                    "predictions": tuple(pending["predictions"]),
                    "targets": tuple(pending["targets"]),
                    "groups": tuple(pending["groups"]),
                }
                for (_condition, metric_id), pending in suite_pending.items()
            },
        )


def _update_digest(digest, value) -> None:
    if value is None:
        digest.update(b"none")
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            _update_digest(digest, value[key])
    elif isinstance(value, (str, int, float, bool)):
        digest.update(repr(value).encode("utf-8"))
    else:
        array = np.asarray(value)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())


def _case_commitment(case) -> str:
    digest = hashlib.sha256()
    for value in (
        case.group_key,
        case.query.observations,
        case.query.actions,
        case.query.horizons,
        case.query.metadata,
        case.targets,
    ):
        _update_digest(digest, value)
    return digest.hexdigest()


def _public_state(value) -> dict[str, object]:
    state = {}
    for key, item in vars(value).items():
        if key in {
            "context",
            "episode",
            "weights",
            "representation",
            "engine",
            "theta",
            "gamma",
            "checkpoint_path",
        }:
            continue
        if isinstance(item, (str, int, float, bool, type(None), tuple, list, dict)):
            state[key] = item
    return state


def _implementation_digest(*values) -> str:
    digest = hashlib.sha256()
    for value in values:
        if inspect.isclass(value) or inspect.isfunction(value) or inspect.ismodule(value):
            digest.update(inspect.getsource(value).encode("utf-8"))
        else:
            try:
                payload = json.dumps(value, sort_keys=True, default=repr)
            except TypeError:
                payload = repr(value)
            digest.update(payload.encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"
