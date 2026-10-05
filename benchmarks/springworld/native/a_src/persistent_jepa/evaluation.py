"""System-level functional and representation evaluation without test leakage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import PersistentJEPA
from .sampling import VALID_ANCHORS, deterministic_derangement
from .torch_data import HORIZONS, PairedBatch, SplitArrays


RIDGE_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclass
class EvalArrays:
    target: PairedBatch
    donor_history_states: torch.Tensor
    donor_history_actions: torch.Tensor
    system_index: np.ndarray
    shuffled_row_index: np.ndarray


def _windows_for_split(data: SplitArrays, seed: int, windows_per_system: int = 8) -> EvalArrays:
    rng = np.random.default_rng(seed)
    systems_count, rollouts = data.states.shape[:2]
    system_index = np.repeat(np.arange(systems_count), windows_per_system)
    count = system_index.size
    target_rollout = rng.integers(rollouts, size=count)
    donor_offset = rng.integers(1, rollouts, size=count)
    donor_rollout = (target_rollout + donor_offset) % rollouts
    target_anchor = rng.choice(VALID_ANCHORS, size=count)
    donor_anchor = rng.choice(VALID_ANCHORS, size=count)

    def history(states: np.ndarray, rollout: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        time = anchor[:, None] + np.arange(-23, 1)[None, :]
        return states[system_index[:, None], rollout[:, None], time]

    def history_actions(actions: np.ndarray, rollout: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        time = anchor[:, None] + np.arange(-23, 0)[None, :]
        return actions[system_index[:, None], rollout[:, None], time]

    target_history = history(data.states, target_rollout, target_anchor)
    target_history_actions = history_actions(data.actions, target_rollout, target_anchor)
    donor_history = history(data.states, donor_rollout, donor_anchor)
    donor_actions = history_actions(data.actions, donor_rollout, donor_anchor)
    target_time = target_anchor[:, None] + np.asarray(HORIZONS)[None, :]
    target_states = data.states[system_index[:, None], target_rollout[:, None], target_time]
    future = np.zeros((count, 3, 16, 2), dtype=np.float32)
    masks = np.zeros((count, 3, 16), dtype=np.float32)
    for hi, horizon in enumerate(HORIZONS):
        time = target_anchor[:, None] + np.arange(horizon)[None, :]
        future[:, hi, :horizon] = data.actions[
            system_index[:, None], target_rollout[:, None], time
        ]
        masks[:, hi, :horizon] = 1.0

    shuffled_system = deterministic_derangement(systems_count, seed + 7919)
    # Rows are system-major, so the same within-system window index is retained.
    shuffled_row_index = (
        shuffled_system[system_index] * windows_per_system
        + np.tile(np.arange(windows_per_system), systems_count)
    )
    target = PairedBatch(
        history_states=torch.from_numpy(np.asarray(target_history).copy()),
        history_actions=torch.from_numpy(np.asarray(target_history_actions).copy()),
        target_states=torch.from_numpy(np.asarray(target_states).copy()),
        future_actions=torch.from_numpy(future),
        action_masks=torch.from_numpy(masks),
        gamma=torch.from_numpy(data.gamma[system_index].copy()),
    )
    return EvalArrays(
        target=target,
        donor_history_states=torch.from_numpy(np.asarray(donor_history).copy()),
        donor_history_actions=torch.from_numpy(np.asarray(donor_actions).copy()),
        system_index=system_index,
        shuffled_row_index=shuffled_row_index,
    )


def state_normalizer(train: SplitArrays) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(train.states).reshape(-1, 4).astype(np.float64)
    return states.mean(axis=0), states.std(axis=0).clip(1e-8)


def system_mean(values: np.ndarray, system_index: np.ndarray, systems: int) -> np.ndarray:
    output = np.empty((systems,) + values.shape[1:], dtype=np.float64)
    for system in range(systems):
        output[system] = values[system_index == system].mean(axis=0)
    return output


class Ridge:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.x_mean: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.weight: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        x64, y64 = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if y64.ndim == 1:
            y64 = y64[:, None]
        self.x_mean, self.y_mean = x64.mean(0), y64.mean(0)
        xc, yc = x64 - self.x_mean, y64 - self.y_mean
        gram = xc.T @ xc + self.alpha * np.eye(xc.shape[1])
        self.weight = np.linalg.solve(gram, xc.T @ yc)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.x_mean is not None and self.y_mean is not None and self.weight is not None
        value = (np.asarray(x, dtype=np.float64) - self.x_mean) @ self.weight + self.y_mean
        return value[:, 0] if value.shape[1] == 1 else value


def mse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(y) - np.asarray(prediction))))


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    y64, p64 = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    denom = np.square(y64 - y64.mean()).sum()
    return float(1.0 - np.square(y64 - p64).sum() / denom)


def select_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    fixed_alpha: float | None = None,
) -> tuple[Ridge, float, float]:
    candidates = (fixed_alpha,) if fixed_alpha is not None else RIDGE_GRID
    scored = []
    for alpha in candidates:
        model = Ridge(float(alpha)).fit(train_x, train_y)
        scored.append((mse(val_y, model.predict(val_x)), float(alpha), model))
    score, alpha, model = min(scored, key=lambda item: (item[0], item[1]))
    return model, alpha, score


@torch.inference_mode()
def extract(
    model: PersistentJEPA,
    arrays: EvalArrays,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    model.eval()
    true_h, representation, predictions = [], [], []
    correct_predictions, shuffled_predictions = [], []
    count = arrays.target.history_states.shape[0]
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        sl = slice(start, stop)
        batch = PairedBatch(**{key: value[sl].to(device) for key, value in arrays.target.__dict__.items()})
        history_h, target_h = model.encode_observations(batch.history_states, batch.target_states)
        z_s, z_p, context = model.codes(history_h, batch.history_actions)
        per_horizon = []
        for hi in range(3):
            horizon_index = torch.full((stop - start,), hi, device=device, dtype=torch.long)
            per_horizon.append(
                model.predictor(
                    context, batch.future_actions[:, hi], batch.action_masks[:, hi], horizon_index
                )
            )
        predictions.append(torch.stack(per_horizon, dim=1).float().cpu())
        true_h.append(target_h.float().cpu())
        representation.append((context if z_p is None else z_p).float().cpu())

        if z_p is not None:
            donor_states = arrays.donor_history_states[sl].to(device)
            donor_actions = arrays.donor_history_actions[sl].to(device)
            donor_h = model.observation(donor_states)
            donor_z = model.persistent(donor_h, donor_actions)
            h16_index = torch.full((stop - start,), 2, device=device, dtype=torch.long)
            correct_context = torch.cat([z_s, donor_z], dim=-1)
            correct_predictions.append(
                model.predictor(
                    correct_context,
                    batch.future_actions[:, 2],
                    batch.action_masks[:, 2],
                    h16_index,
                ).float().cpu()
            )
    output = {
        "true_h": torch.cat(true_h).numpy(),
        "self_prediction_h": torch.cat(predictions).numpy(),
        "representation": torch.cat(representation).numpy(),
        "target_states": arrays.target.target_states.numpy(),
        "gamma": arrays.target.gamma.numpy(),
    }
    if correct_predictions:
        output["correct_prediction_h16"] = torch.cat(correct_predictions).numpy()
        # Correct donor embeddings are already aligned system-major. Applying the
        # frozen row derangement creates a different-system donor without re-encoding.
        output["shuffled_prediction_h16"] = np.empty_like(output["correct_prediction_h16"])
        # Filled by a second predictor pass because donor swapping must happen before P.
        shuffled_rows = arrays.shuffled_row_index
        donor_h_all, donor_z_all, z_s_all = [], [], []
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            target_states = arrays.target.history_states[start:stop].to(device)
            target_actions = arrays.target.history_actions[start:stop].to(device)
            target_h = model.observation(target_states)
            z_s_all.append(model.transient(target_h[:, -1]).float().cpu())
            donor_states = arrays.donor_history_states[start:stop].to(device)
            donor_actions = arrays.donor_history_actions[start:stop].to(device)
            donor_h_all.append(model.observation(donor_states).float().cpu())
            donor_z_all.append(model.persistent(model.observation(donor_states), donor_actions).float().cpu())
        z_s_cpu = torch.cat(z_s_all)
        donor_z_cpu = torch.cat(donor_z_all)
        shuffled_out = []
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            context = torch.cat(
                [z_s_cpu[start:stop].to(device), donor_z_cpu[shuffled_rows[start:stop]].to(device)], dim=-1
            )
            future = arrays.target.future_actions[start:stop, 2].to(device)
            mask = arrays.target.action_masks[start:stop, 2].to(device)
            hi = torch.full((stop - start,), 2, device=device, dtype=torch.long)
            shuffled_out.append(model.predictor(context, future, mask, hi).float().cpu())
        output["shuffled_prediction_h16"] = torch.cat(shuffled_out).numpy()
    return output


class GammaMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def mlp_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    seed: int = 20260819,
    steps: int = 1000,
) -> np.ndarray:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    x_mean, x_std = train_x.mean(0), train_x.std(0).clip(1e-8)
    y_mean, y_std = float(train_y.mean()), float(train_y.std())
    x = torch.from_numpy(((train_x - x_mean) / x_std).astype(np.float32)).to(device)
    y = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32)).to(device)
    model = GammaMLP(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        value = model(torch.from_numpy(((eval_x - x_mean) / x_std).astype(np.float32))).numpy()
    return value * y_std + y_mean
