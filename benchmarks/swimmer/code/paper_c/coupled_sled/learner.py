import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .learner_data import LearnerArrays, build_development_arrays, condition_names
from .manifests import load_spec


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


CANONICAL_ARCHITECTURE_TAG = "concat_mlp_v1"
MASKED_GRU_ARCHITECTURE_TAG = "masked_gru_v1"


class MaskedGRUPersistentAggregator(nn.Module):
    """Two-step recurrent aggregator with an exact identity masked step."""

    def __init__(self, embedding_dim: int, persistent_dim: int):
        super().__init__()
        if embedding_dim % 4:
            raise ValueError("masked-GRU requires history_embedding_dim divisible by four")
        self.embedding_dim = int(embedding_dim)
        self.persistent_dim = int(persistent_dim)
        self.hidden_dim = 5 * self.embedding_dim // 4
        self.cell = nn.GRUCell(self.embedding_dim, self.hidden_dim, bias=True)
        self.projection = nn.Linear(self.hidden_dim, self.persistent_dim, bias=True)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 3 or encoded.shape[1] != 2 or encoded.shape[2] != self.embedding_dim:
            raise ValueError("masked-GRU expects encoded history with shape [batch,2,d_E]")
        if mask.shape != encoded.shape[:2]:
            raise ValueError("masked-GRU history mask shape mismatch")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("masked-GRU history mask must be binary")
        state = torch.zeros(
            encoded.shape[0], self.hidden_dim, dtype=encoded.dtype, device=encoded.device,
        )
        for step in range(2):
            active = mask[:, step].to(dtype=torch.bool)
            if bool(active.any()):
                next_state = state.clone()
                next_state[active] = self.cell(encoded[active, step], state[active])
                state = next_state
        return self.projection(state)


class RawHistoryPredictor(nn.Module):
    def __init__(self, history_dim: int, query_dim: int, hidden_dim: int, target_dim: int):
        super().__init__()
        self.predictor = MLP(2 * history_dim + 2 + query_dim, hidden_dim, target_dim)

    def forward(self, history: torch.Tensor, mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        masked = history * mask[:, :, None]
        return self.predictor(torch.cat((masked.flatten(1), mask, query), dim=1))


class PersistentJEPA(nn.Module):
    def __init__(self, history_dim: int, query_dim: int, target_dim: int, cfg: Dict):
        super().__init__()
        h = cfg["hidden_dim"]
        e = cfg["history_embedding_dim"]
        z = cfg["persistent_dim"]
        q = cfg["query_embedding_dim"]
        aggregator_type = str(cfg.get("aggregator_type", "concat_mlp"))
        if aggregator_type == "concat_mlp":
            self.architecture_tag = CANONICAL_ARCHITECTURE_TAG
        elif aggregator_type == "masked_gru":
            self.architecture_tag = MASKED_GRU_ARCHITECTURE_TAG
        else:
            raise ValueError(f"unknown persistent aggregator type: {aggregator_type}")
        self.segment_encoder = MLP(history_dim, h, e)
        self.aggregator = (
            MLP(2 * e + 2, h, z)
            if self.architecture_tag == CANONICAL_ARCHITECTURE_TAG
            else MaskedGRUPersistentAggregator(e, z)
        )
        self.query_encoder = MLP(query_dim, h, q)
        self.target_encoder = MLP(target_dim, h, z)
        self.latent_predictor = MLP(z + q, h, z)
        self.target_decoder = MLP(z, h, target_dim)

    def persistent(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.segment_encoder(history) * mask[:, :, None]
        return self.aggregate_encoded(encoded, mask)

    def aggregate_encoded(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Aggregate already encoded segments without exposing aggregator internals."""
        if self.architecture_tag == CANONICAL_ARCHITECTURE_TAG:
            return self.aggregator(torch.cat((encoded.flatten(1), mask), dim=1))
        return self.aggregator(encoded, mask)

    def forward(self, history: torch.Tensor, mask: torch.Tensor, query: torch.Tensor, target: torch.Tensor | None = None):
        persistent = self.persistent(history, mask)
        query_embedding = self.query_encoder(query)
        predicted_latent = self.latent_predictor(torch.cat((persistent, query_embedding), dim=1))
        prediction = self.target_decoder(predicted_latent)
        if target is None:
            return prediction, persistent, predicted_latent, None, None
        target_latent = self.target_encoder(target)
        reconstruction = self.target_decoder(target_latent)
        return prediction, persistent, predicted_latent, target_latent, reconstruction


def build_persistent_jepa(
    history_dim: int,
    query_dim: int,
    target_dim: int,
    cfg: Dict,
    expected_architecture_tag: str | None = None,
) -> PersistentJEPA:
    """Single factory for canonical and masked-GRU Persistent-JEPA models."""
    model = PersistentJEPA(history_dim, query_dim, target_dim, cfg)
    if expected_architecture_tag is not None and model.architecture_tag != expected_architecture_tag:
        raise RuntimeError(
            f"architecture tag mismatch: expected {expected_architecture_tag}, got {model.architecture_tag}"
        )
    return model


def tagged_checkpoint_payload(model: PersistentJEPA, dimensions: Dict[str, int]) -> Dict:
    required = {"history_dim", "query_dim", "target_dim"}
    if set(dimensions) != required or any(int(dimensions[key]) <= 0 for key in required):
        raise ValueError("tagged checkpoint dimensions are incomplete")
    return {
        "schema_version": "1.0",
        "architecture_tag": model.architecture_tag,
        "dimensions": {key: int(dimensions[key]) for key in sorted(required)},
        "state_dict": model.state_dict(),
    }


def load_tagged_persistent_jepa(
    checkpoint: str | Path,
    cfg: Dict,
    expected_architecture_tag: str,
    device: torch.device,
) -> PersistentJEPA:
    """Load a new-format checkpoint and fail closed on architecture mismatch."""
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise RuntimeError("tagged Persistent-JEPA checkpoint envelope is missing")
    if payload.get("architecture_tag") != expected_architecture_tag:
        raise RuntimeError("Persistent-JEPA checkpoint architecture tag mismatch")
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, dict):
        raise RuntimeError("Persistent-JEPA checkpoint dimensions are missing")
    model = build_persistent_jepa(
        int(dimensions["history_dim"]), int(dimensions["query_dim"]), int(dimensions["target_dim"]),
        cfg, expected_architecture_tag=expected_architecture_tag,
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model


def _normalizers(train: LearnerArrays) -> Dict[str, np.ndarray]:
    active = train.history[train.history_mask.astype(bool)]
    values = {
        "history_mean": active.mean(axis=0), "history_std": active.std(axis=0),
        "query_mean": train.query_action.mean(axis=0), "query_std": train.query_action.std(axis=0),
        "target_mean": train.target.mean(axis=0), "target_std": train.target.std(axis=0),
    }
    for key in ("history_std", "query_std", "target_std"):
        values[key] = np.maximum(values[key], 1e-6)
    return values


def _normalized(arrays: LearnerArrays, norms: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history = (arrays.history - norms["history_mean"]) / norms["history_std"]
    history *= arrays.history_mask[:, :, None]
    query = (arrays.query_action - norms["query_mean"]) / norms["query_std"]
    target = (arrays.target - norms["target_mean"]) / norms["target_std"]
    return history.astype(np.float32), arrays.history_mask.astype(np.float32), query.astype(np.float32), target.astype(np.float32)


def _loader(arrays: LearnerArrays, norms: Dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [torch.from_numpy(value) for value in _normalized(arrays, norms)]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def _aggregate_mse(model: nn.Module, loader: DataLoader, device: torch.device, jepa: bool) -> float:
    total = 0.0
    count = 0
    for history, mask, query, target in loader:
        history, mask, query, target = (value.to(device) for value in (history, mask, query, target))
        prediction = model(history, mask, query)[0] if jepa else model(history, mask, query)
        total += float(torch.sum((prediction - target) ** 2).cpu())
        count += target.numel()
    return total / count


def _train_raw(model: RawHistoryPredictor, train_loader: DataLoader, select_loader: DataLoader, cfg: Dict, device: torch.device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_loss = float("inf")
    best_state = None
    history_rows = []
    bad_epochs = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for values in train_loader:
            history, mask, query, target = (value.to(device) for value in values)
            loss = torch.mean((model(history, mask, query) - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        selection_loss = _aggregate_mse(model, select_loader, device, False)
        history_rows.append({"epoch": epoch + 1, "selection_mse": selection_loss})
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch + 1 >= cfg.get("minimum_epochs", 1) and bad_epochs >= cfg.get("early_stopping_patience", cfg["epochs"] + 1):
            break
    model.load_state_dict(best_state)
    return history_rows, best_loss


def _train_jepa(model: PersistentJEPA, train_loader: DataLoader, select_loader: DataLoader, cfg: Dict, device: torch.device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_loss = float("inf")
    best_state = None
    history_rows = []
    bad_epochs = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for values in train_loader:
            history, mask, query, target = (value.to(device) for value in values)
            prediction, _, predicted_latent, target_latent, reconstruction = model(history, mask, query, target)
            loss = (
                cfg["jepa_prediction_weight"] * torch.mean((prediction - target) ** 2)
                + cfg["jepa_latent_weight"] * torch.mean((predicted_latent - target_latent.detach()) ** 2)
                + cfg["jepa_target_reconstruction_weight"] * torch.mean((reconstruction - target) ** 2)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        selection_loss = _aggregate_mse(model, select_loader, device, True)
        history_rows.append({"epoch": epoch + 1, "selection_mse": selection_loss})
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch + 1 >= cfg.get("minimum_epochs", 1) and bad_epochs >= cfg.get("early_stopping_patience", cfg["epochs"] + 1):
            break
    model.load_state_dict(best_state)
    return history_rows, best_loss


@torch.no_grad()
def _predictions_and_z(model: nn.Module, arrays: LearnerArrays, norms: Dict[str, np.ndarray], cfg: Dict, device: torch.device, jepa: bool):
    loader = _loader(arrays, norms, cfg["batch_size"], False)
    predictions = []
    embeddings = []
    for history, mask, query, target in loader:
        history, mask, query = (value.to(device) for value in (history, mask, query))
        if jepa:
            prediction, persistent, _, _, _ = model(history, mask, query)
            embeddings.append(persistent.cpu().numpy())
        else:
            prediction = model(history, mask, query)
        predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions), (np.concatenate(embeddings) if embeddings else None)


def _condition_table(arrays: LearnerArrays, prediction: np.ndarray, norms: Dict[str, np.ndarray], model_name: str) -> pd.DataFrame:
    target = (arrays.target - norms["target_mean"]) / norms["target_std"]
    sample_mse = np.mean((prediction - target) ** 2, axis=1)
    names = condition_names()
    rows = []
    rng = np.random.default_rng(44191)
    systems = np.unique(arrays.system_index)
    for condition_index, condition in enumerate(names):
        selected = arrays.condition == condition_index
        per_system = np.asarray([sample_mse[selected & (arrays.system_index == system)].mean() for system in systems])
        boot = per_system[rng.integers(0, len(systems), size=(2000, len(systems)))].mean(axis=1)
        rows.append({
            "model": model_name,
            "condition": condition,
            "mse": float(per_system.mean()),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
        })
    return pd.DataFrame(rows)


def _readout(train_arrays, train_z, holdout_arrays, holdout_z) -> pd.DataFrame:
    names = condition_names()
    parameter_names = ("m_L", "m_R", "b_L", "b_R", "k_c", "d_c")
    rows = []
    for condition_index, condition in enumerate(names):
        if condition in ("query_only", "wrong_system"):
            continue
        train_sel = (train_arrays.condition == condition_index) & (train_arrays.query_index == 0)
        hold_sel = (holdout_arrays.condition == condition_index) & (holdout_arrays.query_index == 0)
        reg = Ridge(alpha=1.0).fit(train_z[train_sel], train_arrays.theta[train_sel])
        prediction = reg.predict(holdout_z[hold_sel])
        scores = r2_score(holdout_arrays.theta[hold_sel], prediction, multioutput="raw_values")
        for name, score in zip(parameter_names, scores):
            rows.append({"condition": condition, "parameter": name, "r2": float(score)})
    return pd.DataFrame(rows)


def _plot_results(condition_table: pd.DataFrame, relation: pd.DataFrame, readout: pd.DataFrame, output_root: Path) -> None:
    order = list(condition_names())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for model, offset, color in (("raw", -0.15, "#777777"), ("jepa", 0.15, "#315a8a")):
        data = condition_table[condition_table.model == model].set_index("condition").loc[order]
        x = np.arange(len(order)) + offset
        axes[0, 0].bar(x, data.mse, width=0.28, label=model, color=color)
        axes[0, 0].errorbar(x, data.mse, yerr=[data.mse - data.ci_low, data.ci_high - data.mse], fmt="none", color="black", capsize=2)
    axes[0, 0].set_xticks(np.arange(len(order)), order, rotation=30, ha="right")
    axes[0, 0].set_ylabel("standardized query MSE")
    axes[0, 0].legend()
    for model, marker, color in (("raw", "o", "#777777"), ("jepa", "s", "#315a8a")):
        data = relation[relation.model == model]
        axes[0, 1].scatter(data.physical_delta_iq, data.learned_marginal_gain, marker=marker, color=color, label=model)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_xlabel("physical marginal information")
    axes[0, 1].set_ylabel("learned marginal gain")
    axes[0, 1].legend()
    pivot = readout.pivot(index="condition", columns="parameter", values="r2")
    image = axes[1, 0].imshow(pivot.values, aspect="auto", vmin=-0.2, vmax=1.0, cmap="viridis")
    axes[1, 0].set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    axes[1, 0].set_yticks(np.arange(len(pivot.index)), pivot.index)
    axes[1, 0].set_title("persistent-z physics readout R2")
    fig.colorbar(image, ax=axes[1, 0])
    summary = condition_table.pivot(index="condition", columns="model", values="mse")
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.95, summary.round(4).to_string(), va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(output_root / "four_core_results.png", dpi=180)
    plt.close(fig)


def run_development_learner(spec_path: Path, manifest_root: Path, construction_path: Path, physical_pairs_path: Path, output_root: Path, device_name: str = "auto") -> Dict:
    spec = load_spec(spec_path)
    cfg = spec["learner_development"]
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    arrays = build_development_arrays(spec_path, manifest_root, construction_path)
    norms = _normalizers(arrays["train"])
    history_dim = arrays["train"].history.shape[2]
    query_dim = arrays["train"].query_action.shape[1]
    target_dim = arrays["train"].target.shape[1]
    train_loader = _loader(arrays["train"], norms, cfg["batch_size"], True)
    select_loader = _loader(arrays["select"], norms, cfg["batch_size"], False)

    raw = RawHistoryPredictor(history_dim, query_dim, cfg["hidden_dim"], target_dim).to(device)
    jepa = PersistentJEPA(history_dim, query_dim, target_dim, cfg).to(device)
    raw_curve, raw_select = _train_raw(raw, train_loader, select_loader, cfg, device)
    jepa_curve, jepa_select = _train_jepa(jepa, train_loader, select_loader, cfg, device)
    output_root.mkdir(parents=True, exist_ok=True)
    torch.save(raw.state_dict(), output_root / "raw_history_development.pt")
    torch.save(jepa.state_dict(), output_root / "persistent_jepa_development.pt")
    pd.DataFrame(raw_curve).assign(model="raw").to_csv(output_root / "raw_training_curve.csv", index=False)
    pd.DataFrame(jepa_curve).assign(model="jepa").to_csv(output_root / "jepa_training_curve.csv", index=False)
    np.savez(output_root / "train_only_normalization.npz", **norms)

    raw_pred, _ = _predictions_and_z(raw, arrays["holdout"], norms, cfg, device, False)
    jepa_pred, holdout_z = _predictions_and_z(jepa, arrays["holdout"], norms, cfg, device, True)
    _, train_z = _predictions_and_z(jepa, arrays["train"], norms, cfg, device, True)
    condition_table = pd.concat([
        _condition_table(arrays["holdout"], raw_pred, norms, "raw"),
        _condition_table(arrays["holdout"], jepa_pred, norms, "jepa"),
    ], ignore_index=True)
    condition_table.to_csv(output_root / "condition_query_prediction.csv", index=False)
    readout = _readout(arrays["train"], train_z, arrays["holdout"], holdout_z)
    readout.to_csv(output_root / "persistent_z_physics_readout.csv", index=False)

    construction = json.loads(construction_path.read_text())
    pair_table = pd.read_csv(physical_pairs_path)
    relation_rows = []
    names = condition_names()
    for model_name, prediction in (("raw", raw_pred), ("jepa", jepa_pred)):
        target = (arrays["holdout"].target - norms["target_mean"]) / norms["target_std"]
        losses = np.mean((prediction - target) ** 2, axis=1)
        for anchor_index, pair in enumerate(construction["pairs"]):
            anchor_loss = losses[(arrays["holdout"].anchor_index == anchor_index) & (arrays["holdout"].condition == names.index("anchor"))].mean()
            for condition in ("repeated", "redundant", "complementary"):
                learned_loss = losses[(arrays["holdout"].anchor_index == anchor_index) & (arrays["holdout"].condition == names.index(condition))].mean()
                added = pair[condition]
                physical = pair_table[(pair_table.base_probe == pair["anchor"]) & (pair_table.added_probe == added)].iloc[0].delta_iq_frac
                relation_rows.append({"model": model_name, "anchor": pair["anchor"], "condition": condition, "physical_delta_iq": float(physical), "learned_marginal_gain": float(anchor_loss - learned_loss)})
    relation = pd.DataFrame(relation_rows)
    relation.to_csv(output_root / "physical_information_vs_learned_gain.csv", index=False)
    _plot_results(condition_table, relation, readout, output_root)

    summary = condition_table.pivot(index="condition", columns="model", values="mse")
    correlations = relation.groupby("model").apply(lambda frame: frame.physical_delta_iq.corr(frame.learned_marginal_gain, method="spearman"), include_groups=False)
    receipt = {
        "status": "DEVELOPMENT_LEARNER_RESULT_NOT_FORMAL",
        "device": device_name,
        "train_systems": cfg["split_system_counts"]["train"],
        "select_systems": cfg["split_system_counts"]["select"],
        "holdout_systems": cfg["split_system_counts"]["holdout"],
        "raw_select_mse": raw_select,
        "jepa_select_mse": jepa_select,
        "raw_comp_minus_red_gain": float(summary.loc["redundant", "raw"] - summary.loc["complementary", "raw"]),
        "jepa_comp_minus_red_gain": float(summary.loc["redundant", "jepa"] - summary.loc["complementary", "jepa"]),
        "raw_physical_gain_spearman": float(correlations["raw"]),
        "jepa_physical_gain_spearman": float(correlations["jepa"]),
        "discovery_accessed": False,
        "design_validation_accessed": False,
        "sealed_accessed": False,
        "formal_learner_pool_generated": False,
    }
    (output_root / "learner_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("construction", type=Path)
    parser.add_argument("physical_pairs", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    args = parser.parse_args()
    print(json.dumps(run_development_learner(args.spec, args.manifest_root, args.construction, args.physical_pairs, args.output_root, args.device), sort_keys=True))


if __name__ == "__main__":
    main()
