"""Frozen R0/PokeWorld bridge evaluation and donor construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .evaluation import RIDGE_GRID, Ridge, r2, select_ridge, system_mean
from .poke_model import PokeJEPA
from .poke_torch import PokeBatch, PokeSplit
from .sampling import deterministic_derangement
from .torch_data import HORIZONS


GAMMA_BINS = ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0000001))


@dataclass
class PokeEvalArrays:
    target: PokeBatch
    correct_donor: PokeBatch
    system_index: np.ndarray
    shuffled_row_index: np.ndarray
    targeted_row_index: dict[str, np.ndarray]
    targeted_diagnostics: dict[str, dict]


def _slice(batch: PokeBatch, start: int, stop: int, device: torch.device) -> PokeBatch:
    return PokeBatch(**{key: value[start:stop].to(device) for key, value in batch.__dict__.items()})


def targeted_systems(data: PokeSplit) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    mass = np.asarray(data.mass, dtype=np.float64)
    gamma = np.asarray(data.gamma, dtype=np.float64)
    stiffness = np.asarray(data.stiffness, dtype=np.float64)
    dm = np.abs(np.log(mass)[:, None] - np.log(mass)[None, :]) / np.log(3.0 / 0.5)
    dk = np.abs(np.log(stiffness)[:, None] - np.log(stiffness)[None, :]) / np.log(6000.0 / 500.0)
    dg = np.abs(gamma[:, None] - gamma[None, :]) / 3.5
    np.fill_diagonal(dm, np.inf)
    np.fill_diagonal(dk, np.inf)
    selected, diagnostics = {}, {}
    for index, (low, high) in enumerate(GAMMA_BINS):
        label = f"bin{index}_{low:.2f}_{high:.2f}"
        output = np.full(mass.shape[0], -1, dtype=np.int64)
        achieved = []
        for target in range(mass.shape[0]):
            valid = (dm[target] <= 0.25) & (dk[target] <= 0.25)
            valid &= (dg[target] >= low) & (dg[target] < high)
            candidates = np.flatnonzero(valid)
            if candidates.size:
                order = np.lexsort((candidates, dm[target, candidates] + dk[target, candidates]))
                donor = int(candidates[order[0]])
                output[target] = donor
                achieved.append((dm[target, donor], dk[target, donor], dg[target, donor]))
        selected[label] = output
        achieved_array = np.asarray(achieved, dtype=np.float64)
        diagnostics[label] = {
            "gamma_bin": [low, min(high, 1.0)],
            "valid_systems": int((output >= 0).sum()),
            "coverage": float((output >= 0).mean()),
            "mean_normalized_log_mass_mismatch": (
                float(achieved_array[:, 0].mean()) if achieved else None
            ),
            "mean_normalized_log_stiffness_mismatch": (
                float(achieved_array[:, 1].mean()) if achieved else None
            ),
            "mean_normalized_gamma_mismatch": (
                float(achieved_array[:, 2].mean()) if achieved else None
            ),
        }
    return selected, diagnostics


def fixed_eval_arrays(
    data: PokeSplit,
    seed: int,
    windows_per_system: int = 8,
    anchor_min: int | None = None,
    include_targeted: bool = True,
) -> PokeEvalArrays:
    rng = np.random.default_rng(seed)
    systems = np.repeat(np.arange(data.states.shape[0]), windows_per_system)
    within = np.tile(np.arange(windows_per_system), data.states.shape[0])
    target_rollout = rng.integers(data.states.shape[1], size=systems.size)
    donor_offset = rng.integers(1, data.states.shape[1], size=systems.size)
    donor_rollout = (target_rollout + donor_offset) % data.states.shape[1]
    anchor_pool = data.anchors[data.anchors >= anchor_min] if anchor_min is not None else data.anchors
    if anchor_pool.size == 0:
        raise ValueError(f"no legal evaluation anchors at or above {anchor_min}")
    target_anchor = rng.choice(anchor_pool, size=systems.size)
    donor_anchor = rng.choice(anchor_pool, size=systems.size)
    target = data._from_indices(systems, target_rollout, target_anchor)
    correct = data._from_indices(systems, donor_rollout, donor_anchor)
    shuffled_system = deterministic_derangement(data.states.shape[0], seed + 7919)
    shuffled_rows = shuffled_system[systems] * windows_per_system + within
    selected, diagnostics = targeted_systems(data) if include_targeted else ({}, {})
    targeted_rows = {
        label: np.where(
            donor_system[systems] >= 0,
            donor_system[systems] * windows_per_system + within,
            -1,
        )
        for label, donor_system in selected.items()
    }
    return PokeEvalArrays(
        target=target,
        correct_donor=correct,
        system_index=systems,
        shuffled_row_index=shuffled_rows,
        targeted_row_index=targeted_rows,
        targeted_diagnostics=diagnostics,
    )


@torch.inference_mode()
def extract_bridge(
    model: PokeJEPA, arrays: PokeEvalArrays, device: torch.device, batch_size: int = 128
) -> dict:
    model.eval()
    count = arrays.system_index.size
    true_h, representation, self_predictions = [], [], []
    z_s_parts, donor_z_parts = [], []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        target = _slice(arrays.target, start, stop, device)
        history_h, target_h = model.encode_batch(target)
        z_s, z_p, context = model.codes(history_h, target.history_actions)
        predictions = []
        for hi in range(3):
            horizon_index = torch.full((stop - start,), hi, device=device, dtype=torch.long)
            predictions.append(
                model.predictor(
                    context, target.future_actions[:, hi], target.action_masks[:, hi], horizon_index
                )
            )
        true_h.append(target_h.float().cpu())
        self_predictions.append(torch.stack(predictions, dim=1).float().cpu())
        representation.append((context if z_p is None else z_p).float().cpu())
        if z_p is not None:
            donor = _slice(arrays.correct_donor, start, stop, device)
            donor_history_h, _ = model.encode_batch(donor)
            donor_z = model.persistent(donor_history_h, donor.history_actions)
            z_s_parts.append(z_s.float().cpu())
            donor_z_parts.append(donor_z.float().cpu())
    output = {
        "true_h": torch.cat(true_h).numpy(),
        "representation": torch.cat(representation).numpy(),
        "self_prediction_h": torch.cat(self_predictions).numpy(),
        "target_states": arrays.target.target_current.numpy(),
    }
    if not z_s_parts:
        return output
    z_s_all, donor_z_all = torch.cat(z_s_parts), torch.cat(donor_z_parts)

    def predict_with_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = np.flatnonzero(rows >= 0)
        predictions = []
        for offset in range(0, valid.size, batch_size):
            target_rows = valid[offset : offset + batch_size]
            context = torch.cat(
                [z_s_all[target_rows].to(device), donor_z_all[rows[target_rows]].to(device)], dim=-1
            )
            future = arrays.target.future_actions[target_rows, 2].to(device)
            mask = arrays.target.action_masks[target_rows, 2].to(device)
            horizon_index = torch.full((target_rows.size,), 2, device=device, dtype=torch.long)
            predictions.append(model.predictor(context, future, mask, horizon_index).float().cpu())
        return (torch.cat(predictions).numpy() if predictions else np.empty((0, 128))), valid

    identity = np.arange(count, dtype=np.int64)
    output["correct_prediction_h16"], _ = predict_with_rows(identity)
    output["shuffled_prediction_h16"], _ = predict_with_rows(arrays.shuffled_row_index)
    output["targeted_prediction_h16"] = {}
    for label, rows in arrays.targeted_row_index.items():
        prediction, valid = predict_with_rows(rows)
        output["targeted_prediction_h16"][label] = {"prediction": prediction, "target_rows": valid}
    return output


def state_normalizers(train: PokeSplit) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    state = np.asarray(train.states, dtype=np.float64).reshape(-1, 8)
    return {
        "object": (state[:, 4:8].mean(0), state[:, 4:8].std(0).clip(1e-8)),
        "full": (state.mean(0), state.std(0).clip(1e-8)),
    }


def normalized_targets(states: np.ndarray, normalizers: dict) -> dict[str, np.ndarray]:
    object_mean, object_std = normalizers["object"]
    full_mean, full_std = normalizers["full"]
    return {
        "object": (states[..., 4:8] - object_mean) / object_std,
        "full": (states - full_mean) / full_std,
    }


def equal_system_components(
    target: np.ndarray, prediction: np.ndarray, system_index: np.ndarray
) -> dict[str, float]:
    squared = np.square(target - prediction)
    systems = np.unique(system_index)
    per_system = np.stack([squared[system_index == system].mean(0) for system in systems])
    dimensions = target.shape[1]
    result = {"state": float(per_system.mean())}
    if dimensions == 4:
        result["position"] = float(per_system[:, :2].mean())
        result["velocity"] = float(per_system[:, 2:].mean())
    return result


def fit_decoder(
    train_h: np.ndarray,
    train_y: np.ndarray,
    eval_h: np.ndarray,
    eval_y: np.ndarray,
    fixed_alpha: float | None = None,
) -> tuple[Ridge, float, float]:
    return select_ridge(train_h, train_y, eval_h, eval_y, fixed_alpha=fixed_alpha)
