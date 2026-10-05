"""Frozen statistical models for the Paper C prospective evidence packet.

This module is deliberately independent of the fresh-outcome evaluator.  It
fits only from explicitly supplied old arrays, scores fresh feature arrays
without targets, and joins realized gains only through a separate function.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROW_KEY = ("system_index", "realization", "history_index", "query_index", "candidate_index")
PAIR_KEY = ("system_index", "realization", "history_index", "query_index")
DEFAULT_LAMBDAS = (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6)
N_HISTORY = 6
N_CANDIDATE = 6


def _as_float_matrix(value: np.ndarray, rows: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[0] != rows:
        raise ValueError(f"{name} must have shape [rows, features]")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def validate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in ROW_KEY if name not in rows]
    if missing:
        raise ValueError(f"prospective rows miss key fields: {missing}")
    result = rows.copy().reset_index(drop=True)
    for name in ROW_KEY:
        result[name] = result[name].astype(np.int64)
    if result.duplicated(list(ROW_KEY)).any():
        raise ValueError("prospective row key is not unique")
    if not result.history_index.between(0, N_HISTORY - 1).all():
        raise ValueError("history_index is outside the frozen six-family vocabulary")
    if not result.query_index.between(0, 5).all():
        raise ValueError("query_index is outside the frozen six-family vocabulary")
    if not result.candidate_index.between(0, N_CANDIDATE - 1).all():
        raise ValueError("candidate_index is outside the frozen six-family vocabulary")
    return result


def _stable_integer(payload: Sequence[object]) -> int:
    encoded = json.dumps(list(payload), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def system_hash_folds(system_index: np.ndarray, salt: int = 79301,
                      folds: int = 5) -> np.ndarray:
    systems = np.asarray(system_index, dtype=np.int64)
    if folds < 2:
        raise ValueError("at least two folds are required")
    mapping = {int(system): _stable_integer((int(system), int(salt))) % folds
               for system in np.unique(systems)}
    return np.asarray([mapping[int(system)] for system in systems], dtype=np.int64)


def coherent_cell_derangement(rows: pd.DataFrame, salt: int) -> np.ndarray:
    """Return a row-position permutation preserving H/Q/candidate and changing system."""
    table = validate_rows(rows)
    table["_position"] = np.arange(len(table), dtype=np.int64)
    donor = np.full(len(table), -1, dtype=np.int64)
    cell = ["history_index", "query_index", "candidate_index"]
    for key, group in table.groupby(cell, sort=True):
        if group.system_index.duplicated().any():
            raise ValueError(f"shuffle cell contains duplicate system rows: {key}")
        systems = np.asarray(sorted(group.system_index.astype(int)), dtype=np.int64)
        if len(systems) < 2:
            raise ValueError(f"shuffle cell contains fewer than two systems: {key}")
        shift = 1 + _stable_integer(("prospective-cell-derangement", int(salt), *map(int, key))) % (len(systems) - 1)
        donor_systems = np.roll(systems, int(shift))
        position = {int(system): int(row_position)
                    for system, row_position in zip(group.system_index, group["_position"])}
        for receiver, source in zip(systems, donor_systems):
            donor[position[int(receiver)]] = position[int(source)]
    if np.any(donor < 0):
        raise RuntimeError("derangement left unmatched rows")
    if np.any(table.system_index.to_numpy()[donor] == table.system_index.to_numpy()):
        raise RuntimeError("derangement retained a physical system")
    for name in cell:
        if not np.array_equal(table[name].to_numpy()[donor], table[name].to_numpy()):
            raise RuntimeError(f"derangement changed {name}")
    if len(np.unique(donor)) != len(donor):
        raise RuntimeError("derangement is not bijective")
    return donor


@dataclass(frozen=True)
class DesignMatrix:
    values: np.ndarray
    continuous: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        continuous = np.asarray(self.continuous, dtype=bool)
        if values.ndim != 2 or continuous.shape != (values.shape[1],):
            raise ValueError("invalid design matrix")
        if len(self.feature_names) != values.shape[1] or not np.isfinite(values).all():
            raise ValueError("invalid design feature metadata")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "continuous", continuous)


def _one_hot(value: np.ndarray, width: int, prefix: str) -> tuple[np.ndarray, list[str]]:
    indices = np.asarray(value, dtype=np.int64)
    if np.any((indices < 0) | (indices >= width)):
        raise ValueError(f"{prefix} index outside frozen vocabulary")
    return np.eye(width, dtype=np.float64)[indices], [f"{prefix}_{i}" for i in range(width)]


def _design(rows: pd.DataFrame, blocks: Sequence[tuple[str, np.ndarray]]) -> DesignMatrix:
    rows = validate_rows(rows)
    history, history_names = _one_hot(rows.history_index, N_HISTORY, "history")
    candidate, candidate_names = _one_hot(rows.candidate_index, N_CANDIDATE, "candidate")
    values, names = [history, candidate], history_names + candidate_names
    continuous = [np.zeros(history.shape[1] + candidate.shape[1], dtype=bool)]
    for prefix, raw in blocks:
        matrix = _as_float_matrix(raw, len(rows), prefix)
        values.append(matrix)
        names.extend([f"{prefix}_{i}" for i in range(matrix.shape[1])])
        continuous.append(np.ones(matrix.shape[1], dtype=bool))
    return DesignMatrix(np.concatenate(values, axis=1), np.concatenate(continuous), tuple(names))


def probe_designs(rows: pd.DataFrame, z_anchor: np.ndarray, delta_persistent: np.ndarray,
                  shuffled_delta: np.ndarray) -> dict[str, DesignMatrix]:
    return {
        "family": _design(rows, ()),
        "anchor": _design(rows, (("z_anchor", z_anchor),)),
        "shuffle": _design(rows, (("z_anchor", z_anchor), ("delta_persistent", shuffled_delta))),
        "true": _design(rows, (("z_anchor", z_anchor), ("delta_persistent", delta_persistent))),
    }


def realization_designs(rows: pd.DataFrame, delta_segment: np.ndarray,
                        delta_persistent: np.ndarray, delta_predicted_query: np.ndarray,
                        eps: float = 1e-12) -> dict[str, DesignMatrix]:
    n = len(rows)
    ds = _as_float_matrix(delta_segment, n, "delta_segment")
    dp = _as_float_matrix(delta_persistent, n, "delta_persistent")
    dq = _as_float_matrix(delta_predicted_query, n, "delta_predicted_query")
    ratio_sp = np.log((np.linalg.norm(dp, axis=1) + eps) / (np.linalg.norm(ds, axis=1) + eps))[:, None]
    ratio_pq = np.log((np.linalg.norm(dq, axis=1) + eps) / (np.linalg.norm(dp, axis=1) + eps))[:, None]
    return {
        "family": _design(rows, ()),
        "upstream": _design(rows, (("delta_segment", ds), ("delta_persistent", dp),
                                    ("log_persistent_over_segment", ratio_sp))),
        "full": _design(rows, (("delta_segment", ds), ("delta_persistent", dp),
                                ("log_persistent_over_segment", ratio_sp),
                                ("delta_predicted_query", dq),
                                ("log_predicted_query_over_persistent", ratio_pq))),
    }


def _fit_scaler(design: DesignMatrix, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(design.values.shape[1], dtype=np.float64)
    scale = np.ones(design.values.shape[1], dtype=np.float64)
    mask = design.continuous
    if np.any(mask):
        mean[mask] = design.values[np.ix_(positions, mask)].mean(axis=0)
        raw = design.values[np.ix_(positions, mask)].std(axis=0, ddof=0)
        scale[mask] = raw
    return mean, scale


def _transform(design: DesignMatrix, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if mean.shape != (design.values.shape[1],) or scale.shape != mean.shape:
        raise ValueError("scaler dimensions do not match design")
    result = design.values.copy()
    mask = design.continuous
    safe_scale = np.where(scale[mask] > 0, scale[mask], 1.0)
    result[:, mask] = (result[:, mask] - mean[mask]) / safe_scale
    result[:, mask & (scale == 0)] = 0.0
    return result


def _system_equal_risk(row_error: np.ndarray, systems: np.ndarray) -> float:
    frame = pd.DataFrame({"system": np.asarray(systems, dtype=np.int64),
                          "error": np.asarray(row_error, dtype=np.float64)})
    return float(frame.groupby("system", sort=True).error.mean().mean())


@dataclass(frozen=True)
class FrozenRidge:
    alpha: float
    feature_names: tuple[str, ...]
    continuous: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    cv_risk: float

    def predict(self, design: DesignMatrix) -> np.ndarray:
        if design.feature_names != self.feature_names or not np.array_equal(design.continuous, self.continuous):
            raise ValueError("fresh design differs from frozen feature order")
        return _transform(design, self.mean, self.scale) @ self.coefficient.T + self.intercept


@dataclass(frozen=True)
class QueryFamilyRidge:
    logical_name: str
    models: Mapping[int, FrozenRidge]
    fold_salt: int
    folds: int
    lambda_grid: tuple[float, ...]

    def predict(self, rows: pd.DataFrame, design: DesignMatrix) -> np.ndarray:
        table = validate_rows(rows)
        widths = {model.coefficient.shape[0] for model in self.models.values()}
        if len(widths) != 1:
            raise RuntimeError("query models have inconsistent output widths")
        result = np.full((len(table), next(iter(widths))), np.nan, dtype=np.float64)
        for query, model in self.models.items():
            positions = np.flatnonzero(table.query_index.to_numpy() == int(query))
            if len(positions):
                subset = DesignMatrix(design.values[positions], design.continuous, design.feature_names)
                result[positions] = model.predict(subset)
        if not np.isfinite(result).all():
            raise ValueError("fresh rows contain an unfitted query family")
        return result


def fit_query_family_ridge(rows: pd.DataFrame, design: DesignMatrix, target: np.ndarray,
                           logical_name: str, fold_salt: int = 79301, folds: int = 5,
                           lambda_grid: Sequence[float] = DEFAULT_LAMBDAS) -> QueryFamilyRidge:
    table = validate_rows(rows)
    y = _as_float_matrix(target, len(table), "target")
    if len(design.values) != len(table):
        raise ValueError("design and rows differ")
    lambdas = tuple(float(value) for value in lambda_grid)
    if not lambdas or any(value < 0 or not np.isfinite(value) for value in lambdas):
        raise ValueError("invalid ridge grid")
    fold = system_hash_folds(table.system_index.to_numpy(), fold_salt, folds)
    fitted: dict[int, FrozenRidge] = {}
    for query in sorted(table.query_index.unique()):
        qpos = np.flatnonzero(table.query_index.to_numpy() == int(query))
        risks: dict[float, float] = {}
        for alpha in lambdas:
            errors, systems = [], []
            for held_out in range(folds):
                train = qpos[fold[qpos] != held_out]
                valid = qpos[fold[qpos] == held_out]
                if not len(train) or not len(valid):
                    raise ValueError(f"empty system fold for query {query}")
                mean, scale = _fit_scaler(design, train)
                reg = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
                reg.fit(_transform(design, mean, scale)[train], y[train])
                prediction = reg.predict(_transform(design, mean, scale)[valid])
                if prediction.ndim == 1:
                    prediction = prediction[:, None]
                errors.append(np.mean((y[valid] - prediction) ** 2, axis=1))
                systems.append(table.system_index.to_numpy()[valid])
            risks[alpha] = _system_equal_risk(np.concatenate(errors), np.concatenate(systems))
        best = min(lambdas, key=lambda alpha: (risks[alpha], -alpha))
        mean, scale = _fit_scaler(design, qpos)
        transformed = _transform(design, mean, scale)
        reg = Ridge(alpha=best, fit_intercept=True, solver="svd").fit(transformed[qpos], y[qpos])
        coefficient = np.asarray(reg.coef_, dtype=np.float64)
        if coefficient.ndim == 1:
            coefficient = coefficient[None, :]
        intercept = np.atleast_1d(np.asarray(reg.intercept_, dtype=np.float64))
        fitted[int(query)] = FrozenRidge(
            best, design.feature_names, design.continuous.copy(), mean, scale,
            coefficient, intercept, risks[best],
        )
    return QueryFamilyRidge(logical_name, fitted, int(fold_salt), int(folds), lambdas)


def fit_probe_models(rows: pd.DataFrame, r_b: np.ndarray, z_anchor: np.ndarray,
                     delta_persistent: np.ndarray, shuffle_salt: int = 79303,
                     fold_salt: int = 79301, folds: int = 5,
                     lambda_grid: Sequence[float] = DEFAULT_LAMBDAS) -> tuple[dict[str, QueryFamilyRidge], np.ndarray]:
    table = validate_rows(rows)
    donor = coherent_cell_derangement(table, shuffle_salt)
    delta = _as_float_matrix(delta_persistent, len(table), "delta_persistent")
    designs = probe_designs(table, z_anchor, delta, delta[donor])
    return ({name: fit_query_family_ridge(table, design, r_b, name, fold_salt, folds, lambda_grid)
             for name, design in designs.items()}, donor)


def predict_probe_models(models: Mapping[str, QueryFamilyRidge], rows: pd.DataFrame,
                         z_anchor: np.ndarray, delta_persistent: np.ndarray,
                         shuffle_salt: int = 79307) -> tuple[dict[str, np.ndarray], np.ndarray]:
    table = validate_rows(rows)
    donor = coherent_cell_derangement(table, shuffle_salt)
    delta = _as_float_matrix(delta_persistent, len(table), "delta_persistent")
    designs = probe_designs(table, z_anchor, delta, delta[donor])
    required = {"family", "anchor", "shuffle", "true"}
    if set(models) != required:
        raise ValueError("probe model set differs from frozen four-model design")
    return ({name: models[name].predict(table, designs[name]) for name in sorted(required)}, donor)


def fit_realization_models(rows: pd.DataFrame, realized_gain: np.ndarray,
                           delta_segment: np.ndarray, delta_persistent: np.ndarray,
                           delta_predicted_query: np.ndarray, fold_salt: int = 79301,
                           folds: int = 5, lambda_grid: Sequence[float] = DEFAULT_LAMBDAS,
                           eps: float = 1e-12) -> dict[str, QueryFamilyRidge]:
    table = validate_rows(rows)
    target = _as_float_matrix(realized_gain, len(table), "realized_gain")
    if target.shape[1] != 1:
        raise ValueError("realization score target must be scalar row gain")
    designs = realization_designs(table, delta_segment, delta_persistent, delta_predicted_query, eps)
    return {name: fit_query_family_ridge(table, design, target, name, fold_salt, folds, lambda_grid)
            for name, design in designs.items()}


def predict_realization_models(models: Mapping[str, QueryFamilyRidge], rows: pd.DataFrame,
                               delta_segment: np.ndarray, delta_persistent: np.ndarray,
                               delta_predicted_query: np.ndarray, permutation_salt: int | None = None,
                               eps: float = 1e-12) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    table = validate_rows(rows)
    required = {"family", "upstream", "full"}
    if set(models) != required:
        raise ValueError("realization model set differs from frozen three-model design")
    designs = realization_designs(table, delta_segment, delta_persistent, delta_predicted_query, eps)
    result = {name: models[name].predict(table, designs[name])[:, 0] for name in sorted(required)}
    donor = None
    if permutation_salt is not None:
        donor = coherent_cell_derangement(table, permutation_salt)
        permuted = realization_designs(
            table, np.asarray(delta_segment)[donor], np.asarray(delta_persistent)[donor],
            np.asarray(delta_predicted_query)[donor], eps,
        )["full"]
        result["permutation"] = models["full"].predict(table, permuted)[:, 0]
    return result, donor


def system_equal_bootstrap(values: np.ndarray, system_index: np.ndarray,
                           replicates: int = 4000, seed: int = 79201) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    systems = np.asarray(system_index, dtype=np.int64)
    if value.ndim != 1 or systems.shape != value.shape or not np.isfinite(value).all():
        raise ValueError("bootstrap requires one finite value per row")
    if int(replicates) < 1:
        raise ValueError("bootstrap requires at least one replicate")
    frame = pd.DataFrame({"system": systems, "value": value})
    per_system = frame.groupby("system", sort=True).value.mean().to_numpy(np.float64)
    if len(per_system) < 2:
        raise ValueError("bootstrap requires at least two physical systems")
    rng = np.random.default_rng(int(seed))
    draws = per_system[rng.integers(0, len(per_system), size=(int(replicates), len(per_system)))].mean(axis=1)
    return {
        "estimate": float(per_system.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "systems": int(len(per_system)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def evaluate_probe_predictions(rows: pd.DataFrame, r_b: np.ndarray,
                               predictions: Mapping[str, np.ndarray],
                               replicates: int = 4000, seed: int = 79501) -> dict[str, dict[str, float | int]]:
    table = validate_rows(rows)
    target = _as_float_matrix(r_b, len(table), "r_b")
    required = {"family", "anchor", "shuffle", "true"}
    if set(predictions) != required:
        raise ValueError("probe predictions differ from frozen model set")
    risk = {}
    for name, prediction in predictions.items():
        pred = _as_float_matrix(prediction, len(table), name)
        if pred.shape != target.shape:
            raise ValueError("probe prediction width differs from r_B")
        risk[name] = np.mean((target - pred) ** 2, axis=1)
    return {
        "delta_anchor": system_equal_bootstrap(risk["anchor"] - risk["true"], table.system_index, replicates, seed),
        "delta_shuffle": system_equal_bootstrap(risk["shuffle"] - risk["true"], table.system_index, replicates, seed),
        "delta_family": system_equal_bootstrap(risk["family"] - risk["true"], table.system_index, replicates, seed),
    }


def _pair_candidate_columns(pairs: pd.DataFrame) -> tuple[str, str]:
    if {"candidate_low", "candidate_high"}.issubset(pairs.columns):
        return "candidate_low", "candidate_high"
    if {"candidate_low_id", "candidate_high_id"}.issubset(pairs.columns):
        return "candidate_low_id", "candidate_high_id"
    raise ValueError("pair manifest lacks candidate_low/candidate_high")


def _tie_sign(row: pd.Series, low_name: str, high_name: str, salt: int) -> int:
    payload = [int(row[name]) for name in PAIR_KEY]
    payload += [int(row[low_name]), int(row[high_name]), int(salt)]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return 1 if int(hashlib.sha256(encoded).hexdigest(), 16) % 2 == 0 else -1


def orient_pairs(rows: pd.DataFrame, scores: np.ndarray, pairs: pd.DataFrame,
                 score_name: str, tie_epsilon: float = 1e-12,
                 tie_salt: int = 79103) -> pd.DataFrame:
    table = validate_rows(rows)
    score = np.asarray(scores, dtype=np.float64)
    if score.shape != (len(table),) or not np.isfinite(score).all():
        raise ValueError("orientation score must be one finite scalar per candidate row")
    indexed = table.copy()
    indexed["_score"] = score
    lookup = indexed.set_index(list(ROW_KEY))._score
    low_name, high_name = _pair_candidate_columns(pairs)
    records = []
    for row in pairs.itertuples(index=False):
        record = row._asdict()
        base = tuple(int(record[name]) for name in PAIR_KEY)
        low, high = int(record[low_name]), int(record[high_name])
        try:
            low_score = float(lookup.loc[base + (low,)])
            high_score = float(lookup.loc[base + (high,)])
        except KeyError as error:
            raise ValueError(f"pair does not join candidate rows: {base + (low, high)}") from error
        difference = low_score - high_score
        tied = abs(difference) <= float(tie_epsilon)
        orientation = _tie_sign(pd.Series(record), low_name, high_name, tie_salt) if tied else (1 if difference > 0 else -1)
        records.append({**{name: int(record[name]) for name in PAIR_KEY},
                        "candidate_low": low, "candidate_high": high,
                        "score_name": str(score_name), "score_low": low_score,
                        "score_high": high_score, "score_difference": difference,
                        "orientation": int(orientation), "score_tie": bool(tied),
                        "tie_epsilon": float(tie_epsilon), "tie_salt": int(tie_salt)})
    return pd.DataFrame(records)


def attach_oriented_gain(rows: pd.DataFrame, realized_gain: np.ndarray,
                         orientation: pd.DataFrame) -> pd.DataFrame:
    table = validate_rows(rows)
    gain = np.asarray(realized_gain, dtype=np.float64)
    if gain.shape != (len(table),) or not np.isfinite(gain).all():
        raise ValueError("realized gain must be one finite scalar per candidate row")
    indexed = table.copy(); indexed["_gain"] = gain
    lookup = indexed.set_index(list(ROW_KEY))._gain
    records = []
    for row in orientation.itertuples(index=False):
        base = tuple(int(getattr(row, name)) for name in PAIR_KEY)
        low_gain = float(lookup.loc[base + (int(row.candidate_low),)])
        high_gain = float(lookup.loc[base + (int(row.candidate_high),)])
        records.append({**row._asdict(), "gain_low": low_gain, "gain_high": high_gain,
                        "oriented_gain": int(row.orientation) * (low_gain - high_gain)})
    return pd.DataFrame(records)


def evaluate_score_orientations(full: pd.DataFrame, family: pd.DataFrame,
                                replicates: int = 4000, seed: int = 79201) -> dict[str, dict[str, float | int]]:
    keys = list(PAIR_KEY) + ["candidate_low", "candidate_high"]
    merged = full.merge(family[keys + ["oriented_gain", "score_tie"]], on=keys, how="inner",
                        validate="one_to_one", suffixes=("_full", "_family"))
    if len(merged) != len(full) or len(merged) != len(family):
        raise ValueError("full and family orientations do not cover identical pairs")
    return {
        "d_full": system_equal_bootstrap(merged.oriented_gain_full, merged.system_index, replicates, seed),
        "d_full_minus_family": system_equal_bootstrap(
            merged.oriented_gain_full - merged.oriented_gain_family,
            merged.system_index, replicates, seed,
        ),
        "tie_fraction_full": {"estimate": float(merged.score_tie_full.mean())},
        "tie_fraction_family": {"estimate": float(merged.score_tie_family.mean())},
    }


def classify_fixed_sequence(primary: Mapping[str, Mapping[str, float]],
                            secondary_a: Mapping[str, Mapping[str, float]],
                            secondary_b: Mapping[str, Mapping[str, float]],
                            underidentified: Sequence[str] = ()) -> dict[str, object]:
    specifications = (
        ("primary_e3", primary, ("route", "specificity")),
        ("secondary_a_decodability", secondary_a, ("delta_anchor", "delta_shuffle")),
        ("secondary_b_realization", secondary_b, ("d_full", "d_full_minus_family")),
    )
    unknown = set(map(str, underidentified))
    allowed = {name for name, _, _ in specifications}
    if not unknown.issubset(allowed):
        raise ValueError(f"unknown fixed-sequence families: {sorted(unknown - allowed)}")
    active, result = True, {"sequence": [name for name, _, _ in specifications], "stopped_at": None}
    for name, values, required in specifications:
        missing = [key for key in required if key not in values or "ci_low" not in values[key]]
        if missing:
            raise ValueError(f"{name} misses interval fields: {missing}")
        passed = False if name in unknown else all(float(values[key]["ci_low"]) > 0 for key in required)
        if active:
            if name in unknown:
                status = "UNDERIDENTIFIED"
                active = False
                result["stopped_at"] = name
            else:
                status = "SUPPORTED" if passed else "NOT_SUPPORTED"
                if not passed:
                    active = False
                    result["stopped_at"] = name
        else:
            status = "EXPLORATORY_AFTER_EARLIER_FAILURE"
        result[name] = {"status": status, "conjunction_passed_nominally": bool(passed)}
    result["family_wise_confirmation_complete"] = result["stopped_at"] is None
    return result


def bundle_digest(bundle: QueryFamilyRidge) -> str:
    digest = hashlib.sha256()
    header = {"logical_name": bundle.logical_name, "fold_salt": bundle.fold_salt,
              "folds": bundle.folds, "lambda_grid": bundle.lambda_grid}
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for query, model in sorted(bundle.models.items()):
        meta = {"query": query, "alpha": model.alpha, "features": model.feature_names,
                "continuous": model.continuous.tolist(), "cv_risk": model.cv_risk}
        digest.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
        for value in (model.mean, model.scale, model.coefficient, model.intercept):
            array = np.ascontiguousarray(value, dtype=np.float64)
            digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def save_bundle(bundle: QueryFamilyRidge, path: Path) -> str:
    """Atomically store a fitted bundle in a non-pickle NPZ artifact."""
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"immutable fitted bundle already exists: {path}")
    arrays: dict[str, np.ndarray] = {}
    metadata = {"logical_name": bundle.logical_name, "fold_salt": bundle.fold_salt,
                "folds": bundle.folds, "lambda_grid": list(bundle.lambda_grid), "models": {}}
    for query, model in sorted(bundle.models.items()):
        prefix = f"q{query}"
        arrays.update({f"{prefix}_mean": model.mean, f"{prefix}_scale": model.scale,
                       f"{prefix}_coefficient": model.coefficient,
                       f"{prefix}_intercept": model.intercept,
                       f"{prefix}_continuous": model.continuous.astype(np.uint8)})
        metadata["models"][str(query)] = {"alpha": model.alpha, "cv_risk": model.cv_risk,
                                            "feature_names": list(model.feature_names)}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)
    return bundle_digest(bundle)


def load_bundle(path: Path) -> QueryFamilyRidge:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        models = {}
        for raw_query, item in metadata["models"].items():
            query, prefix = int(raw_query), f"q{raw_query}"
            models[query] = FrozenRidge(
                float(item["alpha"]), tuple(item["feature_names"]),
                archive[f"{prefix}_continuous"].astype(bool), archive[f"{prefix}_mean"].copy(),
                archive[f"{prefix}_scale"].copy(), archive[f"{prefix}_coefficient"].copy(),
                archive[f"{prefix}_intercept"].copy(), float(item["cv_risk"]),
            )
    return QueryFamilyRidge(metadata["logical_name"], models, int(metadata["fold_salt"]),
                            int(metadata["folds"]), tuple(map(float, metadata["lambda_grid"])))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def old_fit(rows_path: Path, arrays_path: Path, output_dir: Path,
            fold_salt: int = 79301, old_shuffle_salt: int = 79303,
            folds: int = 5, lambda_grid: Sequence[float] = DEFAULT_LAMBDAS) -> dict[str, object]:
    """Fit and serialize every old-system probe and realization model.

    The input NPZ is an explicit old-only handoff.  Requiring both reference
    corrections prevents a later consumer from silently reusing one reference
    stream as its own replication.
    """
    rows_path, arrays_path, output_dir = map(Path, (rows_path, arrays_path, output_dir))
    if output_dir.exists():
        raise RuntimeError(f"old-fit output already exists: {output_dir}")
    rows = validate_rows(pd.read_csv(rows_path))
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    required = {
        "r_b_ref_a", "r_b_ref_b", "z_p_anchor", "delta_segment",
        "delta_persistent", "delta_predicted_query", "realized_gain",
    }
    if set(arrays) != required:
        raise ValueError(f"old-fit arrays must equal {sorted(required)}; got {sorted(arrays)}")
    if any(np.asarray(value).shape[0] != len(rows) for value in arrays.values()):
        raise ValueError("old-fit array row count differs from row manifest")
    output_dir.mkdir(parents=True)
    models: dict[str, QueryFamilyRidge] = {}
    donor_ref = None
    for reference in ("ref_b", "ref_a"):
        fitted, donor = fit_probe_models(
            rows, arrays[f"r_b_{reference}"], arrays["z_p_anchor"], arrays["delta_persistent"],
            old_shuffle_salt, fold_salt, folds, lambda_grid,
        )
        if donor_ref is None:
            donor_ref = donor
        elif not np.array_equal(donor_ref, donor):
            raise RuntimeError("reference streams produced different old shuffle maps")
        models.update({f"probe_{reference}_{name}": model for name, model in fitted.items()})
    realization = fit_realization_models(
        rows, arrays["realized_gain"], arrays["delta_segment"], arrays["delta_persistent"],
        arrays["delta_predicted_query"], fold_salt, folds, lambda_grid,
    )
    models.update({f"realization_{name}": model for name, model in realization.items()})
    artifact_rows = []
    for logical_name, model in sorted(models.items()):
        path = output_dir / f"{logical_name}.npz"
        semantic_sha = save_bundle(model, path)
        artifact_rows.append({"logical_name": logical_name, "path": str(path.resolve()),
                              "file_sha256": sha256_file(path), "semantic_sha256": semantic_sha})
    donor_path = output_dir / "old_probe_shuffle_map.npz"
    with donor_path.open("wb") as handle:
        np.savez_compressed(handle, donor_position=np.asarray(donor_ref, dtype=np.int64),
                            salt=np.asarray(int(old_shuffle_salt), dtype=np.int64))
    manifest = {
        "schema_version": "1.0",
        "status": "PAPER_C_PROSPECTIVE_OLD_FIT_COMPLETE",
        "rows": int(len(rows)),
        "systems": int(rows.system_index.nunique()),
        "row_key": list(ROW_KEY),
        "row_manifest": {"path": str(rows_path.resolve()), "sha256": sha256_file(rows_path)},
        "arrays": {"path": str(arrays_path.resolve()), "sha256": sha256_file(arrays_path)},
        "fold_salt": int(fold_salt), "folds": int(folds),
        "old_shuffle_salt": int(old_shuffle_salt),
        "lambda_grid": list(map(float, lambda_grid)),
        "fresh_data_read": False,
        "formal_models": artifact_rows,
        "old_probe_shuffle_map": {"path": str(donor_path.resolve()), "sha256": sha256_file(donor_path)},
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest_path = output_dir / "OLD_FIT_MANIFEST.json"
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}


def freeze_old_fit(manifest_path: Path, receipt_path: Path,
                   bindings: Sequence[Path] = ()) -> dict[str, object]:
    """Verify an old-fit directory and bind it before any fresh generation."""
    manifest_path, receipt_path = Path(manifest_path), Path(receipt_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "PAPER_C_PROSPECTIVE_OLD_FIT_COMPLETE" or manifest.get("fresh_data_read") is not False:
        raise RuntimeError("old-fit manifest is not eligible for freeze")
    for item in manifest["formal_models"]:
        if sha256_file(Path(item["path"])) != item["file_sha256"]:
            raise RuntimeError(f"old-fit model is stale: {item['logical_name']}")
        if bundle_digest(load_bundle(Path(item["path"]))) != item["semantic_sha256"]:
            raise RuntimeError(f"old-fit model semantic digest is stale: {item['logical_name']}")
    shuffle = manifest["old_probe_shuffle_map"]
    if sha256_file(Path(shuffle["path"])) != shuffle["sha256"]:
        raise RuntimeError("old-fit shuffle map is stale")
    bound = {}
    for path in map(Path, bindings):
        if not path.is_file():
            raise FileNotFoundError(path)
        bound[str(path.resolve())] = sha256_file(path)
    receipt = {
        "schema_version": "1.0",
        "status": "PAPER_C_PROSPECTIVE_OLD_FIT_FROZEN",
        "old_fit_manifest": str(manifest_path.resolve()),
        "old_fit_manifest_sha256": sha256_file(manifest_path),
        "bindings": bound,
        "fresh_generation_authorized": True,
        "fresh_outcome_read": False,
    }
    _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_sha256": sha256_file(receipt_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("old-fit")
    fit.add_argument("--rows", type=Path, required=True)
    fit.add_argument("--arrays", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--fold-salt", type=int, default=79301)
    fit.add_argument("--shuffle-salt", type=int, default=79303)
    freeze = sub.add_parser("freeze-old-fit")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--receipt", type=Path, required=True)
    freeze.add_argument("--bind", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.command == "old-fit":
        result = old_fit(args.rows, args.arrays, args.out, args.fold_salt, args.shuffle_salt)
    else:
        result = freeze_old_fit(args.manifest, args.receipt, args.bind)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
