from __future__ import annotations

import glob
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_H5PY_IMPORT_ERROR = None
try:
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    h5py = None
    _H5PY_IMPORT_ERROR = exc


def _require_h5py() -> None:
    if h5py is None:
        raise RuntimeError(
            "h5py is required for Burgers dataloading. "
            "Install with: pip install h5py"
        ) from _H5PY_IMPORT_ERROR


class L2RELoss(nn.Module):
    def __init__(self):
        super(L2RELoss, self).__init__()

    def forward(self, prediction, ground_truth):
        numerator = torch.norm(prediction - ground_truth, p=2)
        denominator = torch.norm(ground_truth, p=2)
        epsilon = 1e-8
        return numerator / (denominator + epsilon)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    # allows counting for complex weights / for TFNO-like modules
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters() if p.requires_grad]
    )


def sample_translation_batch(
    target_seq: torch.Tensor,   # [B, K, 1, N] — full trajectory per case
    t_idx: torch.Tensor,        # [B, K]       — absolute frame indices (e.g., 0,10,...,1000)
    num_samples: int,           # S — number of (start, end) pairs per case in this batch
    max_horizon: int,           # H — current curriculum max (in frames; offset ∈ [0, H])
    t_norm_denom: float,        # divisor that maps t_idx differences to normalized time
    offset_dist: str = "uniform",  # 'uniform' or 'smallk' (Beta(1,3) biased toward small offsets)
):
    """Sample (start_idx, end_idx) pairs per example for time-translation training.

    offset_dist:
      - 'uniform': offset ~ Uniform{0, 1, ..., H}. E[offset] = H/2.
      - 'smallk':  offset ~ round((H+1) * Beta(1,3)).clamp(0, H). E[offset] ≈ H/4.
                   Triples 1-step sample density vs uniform — useful when AR-style
                   per-step accuracy is the bottleneck.

    Returns:
      ic        [B*S, 1, N]  — target_seq[b, start_idx[b,s]]   (treated as new IC)
      target    [B*S, 1, N]  — target_seq[b, end_idx[b,s]]     (supervision)
      t_rel     [B*S, 1]     — (t_idx[b, end] - t_idx[b, start]) / t_norm_denom
    """
    if target_seq.ndim != 4:
        raise ValueError(f"target_seq must be [B,K,1,N], got {tuple(target_seq.shape)}")
    if t_idx.ndim != 2 or t_idx.shape[:2] != target_seq.shape[:2]:
        raise ValueError(f"t_idx shape mismatch with target_seq")

    b, k, _, n = target_seq.shape
    h = int(max_horizon)
    if h < 0 or h >= k:
        raise ValueError(f"max_horizon must be in [0, K-1={k-1}], got {h}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    s = int(num_samples)
    device = target_seq.device

    # start_idx ∈ [0, K-1-h], offset ∈ [0, h]; end_idx = start_idx + offset
    start_idx = torch.randint(0, k - h, (b, s), device=device)
    if offset_dist == "uniform":
        offset = torch.randint(0, h + 1, (b, s), device=device)
    elif offset_dist == "smallk":
        # Beta(1, 3) inverse CDF: 1 - (1-U)^(1/3) ∈ [0, 1] with mean 1/4.
        u = torch.rand((b, s), device=device, dtype=target_seq.dtype)
        beta = 1.0 - (1.0 - u).pow(1.0 / 3.0)
        offset = (beta * (h + 1)).long().clamp_(0, h)
    else:
        raise ValueError(f"offset_dist must be 'uniform' or 'smallk', got '{offset_dist}'")
    end_idx   = start_idx + offset

    batch_idx = torch.arange(b, device=device).view(b, 1).expand(b, s)
    ic     = target_seq[batch_idx, start_idx]   # [B, S, 1, N]
    target = target_seq[batch_idx, end_idx]     # [B, S, 1, N]

    t_start = t_idx[batch_idx, start_idx].to(target_seq.dtype)
    t_end   = t_idx[batch_idx, end_idx].to(target_seq.dtype)
    t_rel   = (t_end - t_start) / float(t_norm_denom)  # [B, S]

    ic_flat     = ic.reshape(b * s, 1, n)
    target_flat = target.reshape(b * s, 1, n)
    t_rel_flat  = t_rel.reshape(b * s, 1)
    return ic_flat, target_flat, t_rel_flat


def curriculum_max_horizon(
    epoch: int,                # 1-based current epoch
    warmup_epochs: int,        # epochs to ramp from 1 to k_max
    k_max: int,                # target max horizon (e.g., 100 for 101-frame trajectories)
    schedule: str = "linear",  # 'linear' or 'off' (off = always k_max)
    h_init: int = 1,
) -> int:
    """Linear curriculum: H grows from h_init at epoch 1 to k_max at warmup_epochs, then constant."""
    if schedule == "off" or warmup_epochs <= 1:
        return int(k_max)
    e = max(1, int(epoch))
    if e >= warmup_epochs:
        return int(k_max)
    frac = (e - 1) / max(1, warmup_epochs - 1)
    h = h_init + frac * (k_max - h_init)
    return int(max(h_init, min(k_max, round(h))))


class BurgersPairedDataset(Dataset):
    """
    Burgers dataset with paired conditioning/prediction samples.

    For each sample, two temporally downsampled full-trajectory trunks are
    sampled from the same viscosity shard (same nu):
    - conditioner input: cond trunk -> [K, 401], where K=101 from full [1001]
    - predictor trunk (different random IC/trunk, same nu): pred trunk -> [K, 401]
      - predictor IC: first frame of pred trunk -> [401]
      - target sequence: full pred trunk -> [K, 401]
    """

    SPLIT_CASES = {
        "train": (0, 39),
        "eval": (40, 44),
        "test": (45, 49),
    }

    def __init__(
        self,
        data_root: str = "data/burgers/datagen_python/output_new/task1_viscous_main",
        split: str = "train",
        # The released Burgers shards are already on the 401-point grid.
        # Keep every spatial point; the older 4001-point release used stride 10.
        space_stride: int = 1,
        include_output_add_train: bool = False,
        output_add_root: Optional[str] = None,
        prediction_horizon: int = 101,
        cache_mode: str = "cond_init",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._h5_handles: Dict[int, Any] = {}
        _require_h5py()

        if split not in self.SPLIT_CASES:
            raise ValueError(f"Unsupported split '{split}'. Use one of {list(self.SPLIT_CASES)}.")
        if space_stride <= 0:
            raise ValueError("space_stride must be positive.")
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive.")
        if cache_mode not in {"none", "cond_init", "full"}:
            raise ValueError("cache_mode must be one of {'none', 'cond_init', 'full'}.")

        self.data_root = data_root
        self.split = split
        self.space_stride = int(space_stride)
        self.include_output_add_train = bool(include_output_add_train)
        self.output_add_root = output_add_root
        self.prediction_horizon = int(prediction_horizon)
        self.trunk_len = self.prediction_horizon
        self.cache_mode = cache_mode
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}

        self.shards = self._discover_shards()
        self.num_shards = len(self.shards)
        self.sample_index = self._build_sample_index()
        self.source_sample_counts = self._build_source_sample_counts()
        self.source_shard_counts = self._build_source_shard_counts()
        self.n_t = self.shards[0]["n_t"]
        self.n_x = self.shards[0]["n_x"]
        if self.n_t < 2:
            raise RuntimeError(f"Expected at least 2 time frames, got n_t={self.n_t}.")
        # Fixed temporal contract: no random start, use globally downsampled timeline.
        self.temporal_idx = np.linspace(0, self.n_t - 1, num=self.prediction_horizon, dtype=np.int64)
        if len(np.unique(self.temporal_idx)) != self.prediction_horizon:
            raise RuntimeError(
                "Temporal downsampling produced duplicate indices. "
                f"n_t={self.n_t}, prediction_horizon={self.prediction_horizon}."
            )
        self.trunk_len = int(self.temporal_idx.shape[0])

        # Locked contract for conditioner/predictor input:
        # trunk_ds [K, 401]
        self.expected_cond_t = self.trunk_len
        self.expected_cond_x = (self.n_x - 1) // self.space_stride + 1

        # For full cache mode, preload all shards to CPU RAM up front
        # so __getitem__ performs no HDF5 reads.
        if self.cache_mode == "full":
            self._warmup_full_cache()

    def _discover_shards(self) -> list[dict[str, Any]]:
        roots: list[tuple[str, str]] = [("base", self.data_root)]
        if self.split == "train" and self.include_output_add_train:
            add_root = self.output_add_root or "data/burgers/datagen_python/output_add/task1_viscous_main"
            roots.append(("add", add_root))

        shards: list[dict[str, Any]] = []
        n_t_ref: Optional[int] = None
        n_x_ref: Optional[int] = None
        nu_pattern = re.compile(r"viscous_nu_([0-9]*\.?[0-9]+)\.h5$")

        for source, root in roots:
            pattern = os.path.join(root, "viscous_nu_*.h5")
            paths = sorted(glob.glob(pattern))
            if not paths:
                raise FileNotFoundError(
                    f"No viscous Burgers shards found under '{root}' with pattern '{pattern}'."
                )

            for path in paths:
                with h5py.File(path, "r") as f:
                    if "U_All" not in f:
                        raise ValueError(f"Shard '{path}' does not contain required dataset U_All.")
                    u_shape = tuple(f["U_All"].shape)
                    if len(u_shape) != 3:
                        raise ValueError(f"Shard '{path}' has unsupported U_All shape: {u_shape}.")

                    n_t, n_x, n_cases = int(u_shape[0]), int(u_shape[1]), int(u_shape[2])

                    if n_t_ref is None:
                        n_t_ref, n_x_ref = n_t, n_x
                    else:
                        if n_t != n_t_ref or n_x != n_x_ref:
                            raise ValueError(
                                "All shards must share the same [time, space] dimensions. "
                                f"Expected [{n_t_ref}, {n_x_ref}], got [{n_t}, {n_x}] in '{path}'."
                            )

                    if source == "base":
                        allowed_cases = self._build_base_split_case_indices()
                    else:
                        allowed_cases = list(range(50))

                    max_case = max(allowed_cases)
                    if n_cases <= max_case:
                        raise ValueError(
                            f"Shard '{path}' has {n_cases} cases, but split '{self.split}' requires "
                            f"index <= {max_case}."
                        )

                    filename = Path(path).name
                    match = nu_pattern.match(filename)
                    nu_value = float(f.attrs.get("nu", np.nan))
                    if np.isnan(nu_value):
                        if match is None:
                            raise ValueError(f"Could not infer viscosity nu from '{filename}'.")
                        nu_value = float(match.group(1))

                    shards.append(
                        {
                            "path": path,
                            "source": source,
                            "nu": float(nu_value),
                            "n_t": n_t,
                            "n_x": n_x,
                            "n_cases": n_cases,
                            "allowed_cases": [int(x) for x in allowed_cases],
                        }
                    )

        nu_values = sorted({round(float(shard["nu"]), 12) for shard in shards})
        nu_to_id = {nu: idx for idx, nu in enumerate(nu_values)}
        for shard in shards:
            shard["nu_id"] = int(nu_to_id[round(float(shard["nu"]), 12)])

        return shards

    def _build_base_split_case_indices(self) -> list[int]:
        start, end = self.SPLIT_CASES[self.split]
        return list(range(start, end + 1))

    def _build_sample_index(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for shard_idx, shard in enumerate(self.shards):
            for cond_case_idx in shard["allowed_cases"]:
                pairs.append((int(shard_idx), int(cond_case_idx)))
        return pairs

    def _build_source_sample_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for shard in self.shards:
            src = str(shard["source"])
            counts[src] = counts.get(src, 0) + len(shard["allowed_cases"])
        return counts

    def _build_source_shard_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for shard in self.shards:
            src = str(shard["source"])
            counts[src] = counts.get(src, 0) + 1
        return counts

    def _sample_case_index(self, allowed_cases: list[int], exclude_case: Optional[int] = None) -> int:
        if not allowed_cases:
            raise RuntimeError("No allowed case indices available for sampling.")
        if exclude_case is None or len(allowed_cases) == 1:
            pick = int(self.rng.integers(0, len(allowed_cases)))
            return int(allowed_cases[pick])

        candidates = [c for c in allowed_cases if int(c) != int(exclude_case)]
        if not candidates:
            candidates = allowed_cases
        pick = int(self.rng.integers(0, len(candidates)))
        return int(candidates[pick])

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "num_shards": self.num_shards,
            "num_samples": len(self.sample_index),
            "n_t": self.n_t,
            "n_x": self.n_x,
            "source_shard_counts": dict(self.source_shard_counts),
            "source_sample_counts": dict(self.source_sample_counts),
            "space_stride": self.space_stride,
            "prediction_horizon": self.prediction_horizon,
            "trunk_len": self.trunk_len,
            "cache_mode": self.cache_mode,
            "num_cached_shards": len(self._cache),
            "temporal_index_mode": "fixed_full_trajectory_downsample",
            "temporal_indices_head": [int(x) for x in self.temporal_idx[: min(8, len(self.temporal_idx))]],
            "temporal_indices_tail": [int(x) for x in self.temporal_idx[-min(8, len(self.temporal_idx)) :]],
            "expected_cond_shape": [1, self.expected_cond_t, self.expected_cond_x],
            "expected_pred_u0_shape": [1, self.expected_cond_x],
            "expected_target_shape": [self.prediction_horizon, 1, self.expected_cond_x],
            "expected_target_ds_shape": [self.prediction_horizon, 1, self.expected_cond_x],
            "expected_trunk_shape": [1, self.trunk_len, self.expected_cond_x],
            "expected_t_idx_shape": [self.trunk_len],
        }

    def _get_h5(self, shard_idx: int):
        handle = self._h5_handles.get(shard_idx)
        if handle is None:
            handle = h5py.File(self.shards[shard_idx]["path"], "r")
            self._h5_handles[shard_idx] = handle
        return handle

    def _get_cache(self, shard_idx: int) -> Dict[str, np.ndarray]:
        cache = self._cache.get(shard_idx)
        if cache is not None:
            return cache

        cache = {}
        h5f = self._get_h5(shard_idx)
        if self.cache_mode in {"cond_init", "full"}:
            cache["u_ds"] = np.asarray(
                h5f["U_All"][self.temporal_idx, :: self.space_stride, :],
                dtype=np.float32,
            )  # [K, 401, n_cases]

        self._cache[shard_idx] = cache
        return cache

    def _close_h5_handles(self) -> None:
        for handle in self._h5_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_handles = {}

    def _warmup_full_cache(self) -> None:
        for shard_idx in range(self.num_shards):
            self._get_cache(shard_idx)
            print(f"load {self.shards[shard_idx]['path']}", flush=True)
        # After full preload, file handles are no longer needed.
        self._close_h5_handles()

    def _load_trunk_ds(
        self,
        shard_idx: int,
        case_idx: int,
    ) -> np.ndarray:
        if self.cache_mode == "none":
            h5f = self._get_h5(shard_idx)
            u_all = h5f["U_All"]
            return np.asarray(
                u_all[self.temporal_idx, :: self.space_stride, case_idx],
                dtype=np.float32,
            )  # [K, 401]

        cache = self._get_cache(shard_idx)
        return np.asarray(cache["u_ds"][:, :, case_idx], dtype=np.float32)  # [K, 401]

    def close(self) -> None:
        self._cache = {}
        self._close_h5_handles()

    def __del__(self) -> None:  # pragma: no cover - destructor path
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5_handles"] = {}
        state["_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._h5_handles = {}
        self._cache = {}

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index out of range: {index}")

        shard_idx, cond_case_idx = self.sample_index[index]
        shard = self.shards[shard_idx]
        pred_case_idx = self._sample_case_index(shard["allowed_cases"], exclude_case=cond_case_idx)

        cond_t_idx = self.temporal_idx.copy()
        pred_t_idx = self.temporal_idx.copy()

        cond_np = self._load_trunk_ds(
            shard_idx=shard_idx,
            case_idx=cond_case_idx,
        )  # [K, 401]
        pred_trunk_np = self._load_trunk_ds(
            shard_idx=shard_idx,
            case_idx=pred_case_idx,
        )  # [K, 401]

        pred_u0_np = pred_trunk_np[0]  # [401]
        target_seq_np = pred_trunk_np  # [K, 401]

        if cond_np.shape != (self.expected_cond_t, self.expected_cond_x):
            raise RuntimeError(
                "Conditioner downsample shape mismatch. "
                f"Expected {(self.expected_cond_t, self.expected_cond_x)}, got {cond_np.shape}."
            )
        if pred_u0_np.shape[0] != self.expected_cond_x:
            raise RuntimeError(
                "Predictor downsampled shape mismatch. "
                f"Expected {self.expected_cond_x}, got pred_u0={pred_u0_np.shape}."
            )
        if target_seq_np.shape != (self.prediction_horizon, self.expected_cond_x):
            raise RuntimeError(
                "Future-sequence target shape mismatch. "
                f"Expected {(self.prediction_horizon, self.expected_cond_x)}, got {target_seq_np.shape}."
            )

        sample = {
            "cond_u": torch.from_numpy(cond_np).unsqueeze(0),  # [1, K, 401]
            "pred_u0": torch.from_numpy(pred_u0_np).unsqueeze(0),  # [1, 401]
            "target_seq": torch.from_numpy(target_seq_np).unsqueeze(1),  # [K, 1, 401]
            "trunk_u": torch.from_numpy(pred_trunk_np).unsqueeze(0),  # [1, K, 401] (predictor trunk)
            "cond_trunk_u": torch.from_numpy(cond_np).unsqueeze(0),  # [1, K, 401]
            "pred_trunk_u": torch.from_numpy(pred_trunk_np).unsqueeze(0),  # [1, K, 401]
            "nu_value": torch.tensor([shard["nu"]], dtype=torch.float32),  # [1]
            "nu_id": int(shard["nu_id"]),
            "cond_case_idx": int(cond_case_idx),
            "pred_case_idx": int(pred_case_idx),
            "cond_start_t_idx": 0,
            "pred_start_t_idx": 0,
            "start_t_idx": 0,  # backward-compatible alias
            "cond_t_idx": torch.from_numpy(cond_t_idx.astype(np.int64)),  # [K]
            "pred_t_idx": torch.from_numpy(pred_t_idx.astype(np.int64)),  # [K]
            "t_idx": torch.from_numpy(pred_t_idx.astype(np.int64)),  # [K], backward-compatible alias
        }
        return sample


TRAIN_NUS_BURGERS = (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


class _BurgersGroupedEval(Dataset):
    """In-memory eval set sourced from output_true, filtered to one ν-group:
        - 'id': trained ν shards, all cases (held-out from training-data dir)
        - 'ood': untrained-but-viscous ν shards, all cases
        - 'ood_inviscid': inviscid shard (ν=0), all cases.

    Pre-loads all selected shards into RAM (small: ~100 cases × 9 ν × 101 × 401
    floats ≈ 30 MB per group). Sample dict matches BurgersPairedDataset, so it
    drops into existing run_epoch loops with no other changes.
    """

    def __init__(
        self,
        group: str,
        eval_root: str = "data/burgers/datagen_python/output_true",
        prediction_horizon: int = 101,
        space_stride: int = 10,
        max_cases_per_shard: int = 0,  # 0 = all
        total_max_samples: int = 0,    # 0 = no cap. Uses round-robin to spread across shards.
        deterministic_pairing: bool = False,  # if True, pred_idx = (cond_idx + 1) % n_cases
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        _require_h5py()
        if group not in {"id", "ood", "ood_inviscid"}:
            raise ValueError(f"group must be 'id'/'ood'/'ood_inviscid', got '{group}'.")
        self.group = group
        self.eval_root = eval_root
        self.prediction_horizon = int(prediction_horizon)
        self.space_stride = int(space_stride)
        self.deterministic_pairing = bool(deterministic_pairing)
        self.rng = np.random.default_rng(seed)

        train_set = {round(float(v), 12) for v in TRAIN_NUS_BURGERS}
        nu_pattern = re.compile(r"viscous_nu_([0-9]*\.?[0-9]+)\.h5$")

        shard_paths: list[tuple[str, float]] = []
        if group in {"id", "ood"}:
            for fp in sorted(Path(eval_root).glob("viscous_nu_*.h5")):
                m = nu_pattern.search(fp.name)
                if m is None:
                    continue
                nu = float(m.group(1))
                is_trained = round(nu, 12) in train_set
                if (group == "id" and is_trained) or (group == "ood" and not is_trained):
                    shard_paths.append((str(fp), nu))
        else:  # ood_inviscid
            for fp in sorted(Path(eval_root).glob("inviscid_nu_*.h5")):
                shard_paths.append((str(fp), 0.0))

        if not shard_paths:
            raise RuntimeError(f"No shards found for group='{group}' under '{eval_root}'.")

        # Load all shards. output_true is already at the (101, 401) target grid;
        # if the file is at the full (1001, 4001), downsample by space_stride and
        # an even temporal index.
        self.shards: list[dict[str, Any]] = []
        nu_values = sorted({round(float(nu), 12) for _, nu in shard_paths})
        nu_to_id = {nu: idx for idx, nu in enumerate(nu_values)}
        for path, nu in shard_paths:
            with h5py.File(path, "r") as f:
                arr = np.asarray(f["U_All"][:], dtype=np.float32)  # [n_t, n_x, n_cases]
            n_t, n_x, n_cases = arr.shape
            if n_t != self.prediction_horizon:
                t_idx = np.linspace(0, n_t - 1, num=self.prediction_horizon, dtype=np.int64)
                arr = arr[t_idx]
            target_x = (n_x - 1) // self.space_stride + 1 if n_x > 401 else n_x
            if arr.shape[1] != target_x:
                x_idx = np.arange(0, arr.shape[1], self.space_stride, dtype=np.int64)
                arr = arr[:, x_idx]
            n_use = n_cases if max_cases_per_shard <= 0 else min(int(max_cases_per_shard), n_cases)
            self.shards.append({
                "nu": float(nu),
                "nu_id": int(nu_to_id[round(float(nu), 12)]),
                "data": arr[..., :n_use],  # [pred_T, pred_X, n_use]
                "n_cases": n_use,
            })

        self.expected_cond_t = self.prediction_horizon
        self.expected_cond_x = (self.shards[0]["data"].shape[1])

        # Flat sample list. Round-robin across shards so a `total_max_samples`
        # cap spreads the kept samples evenly across shards rather than
        # truncating the last few shards.
        max_per = max((s["n_cases"] for s in self.shards), default=0)
        self._samples: list[tuple[int, int]] = []
        for c_i in range(max_per):
            for s_i, s in enumerate(self.shards):
                if c_i < s["n_cases"]:
                    self._samples.append((s_i, c_i))
        if total_max_samples > 0:
            self._samples = self._samples[: int(total_max_samples)]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        s_i, cond_idx = self._samples[index]
        s = self.shards[s_i]
        n_cases = s["n_cases"]
        if self.deterministic_pairing and n_cases > 1:
            pred_idx = (cond_idx + 1) % n_cases
        elif n_cases > 1:
            pred_idx = int(self.rng.integers(0, n_cases))
            while pred_idx == cond_idx:
                pred_idx = int(self.rng.integers(0, n_cases))
        else:
            pred_idx = cond_idx
        cond_np = s["data"][..., cond_idx]   # [pred_T, pred_X]
        pred_np = s["data"][..., pred_idx]   # [pred_T, pred_X]
        pred_u0_np = pred_np[0]
        target_seq_np = pred_np
        t_idx = np.arange(self.prediction_horizon, dtype=np.int64)
        return {
            "cond_u": torch.from_numpy(np.ascontiguousarray(cond_np)).unsqueeze(0),
            "pred_u0": torch.from_numpy(np.ascontiguousarray(pred_u0_np)).unsqueeze(0),
            "target_seq": torch.from_numpy(np.ascontiguousarray(target_seq_np)).unsqueeze(1),
            "trunk_u": torch.from_numpy(np.ascontiguousarray(pred_np)).unsqueeze(0),
            "cond_trunk_u": torch.from_numpy(np.ascontiguousarray(cond_np)).unsqueeze(0),
            "pred_trunk_u": torch.from_numpy(np.ascontiguousarray(pred_np)).unsqueeze(0),
            "nu_value": torch.tensor([s["nu"]], dtype=torch.float32),
            "nu_id": int(s["nu_id"]),
            "cond_case_idx": int(cond_idx),
            "pred_case_idx": int(pred_idx),
            "cond_start_t_idx": 0,
            "pred_start_t_idx": 0,
            "start_t_idx": 0,
            "cond_t_idx": torch.from_numpy(t_idx),
            "pred_t_idx": torch.from_numpy(t_idx),
            "t_idx": torch.from_numpy(t_idx),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "num_shards": len(self.shards),
            "num_samples": len(self._samples),
            "nus": [s["nu"] for s in self.shards],
            "cases_per_shard": [s["n_cases"] for s in self.shards],
        }


def create_burgers_dataloaders(
    data_root: str = "data/burgers/datagen_python/output_new/task1_viscous_main",
    include_output_add_train: bool = False,
    output_add_root: str = "data/burgers/datagen_python/output_add/task1_viscous_main",
    prediction_horizon: int = 101,
    cache_mode: str = "cond_init",
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    train_shuffle: bool = True,
    drop_last_train: bool = False,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: int = 4,
    seed: Optional[int] = None,
    include_test: bool = True,) -> dict[str, DataLoader]:
    """
    Factory for Burgers train/eval/test dataloaders.

    ``include_test=False`` is used by the clean training fork so that final
    test/OOD data are not opened during training or validation.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive.")
    if cache_mode == "full" and num_workers > 0:
        print("create_burgers_dataloaders: cache_mode=full requires num_workers=0; overriding.")
        num_workers = 0
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    datasets = {
        "train": BurgersPairedDataset(
            data_root=data_root,
            split="train",
            include_output_add_train=include_output_add_train,
            output_add_root=output_add_root,
            prediction_horizon=prediction_horizon,
            cache_mode=cache_mode,
            seed=seed,
        ),
        "eval": BurgersPairedDataset(
            data_root=data_root,
            split="eval",
            include_output_add_train=False,
            output_add_root=output_add_root,
            prediction_horizon=prediction_horizon,
            cache_mode=cache_mode,
            seed=seed,
        ),
        **({"test": BurgersPairedDataset(
            data_root=data_root,
            split="test",
            include_output_add_train=False,
            output_add_root=output_add_root,
            prediction_horizon=prediction_horizon,
            cache_mode=cache_mode,
            seed=seed,
        )} if include_test else {}),
    }

    train_kwargs: Dict[str, Any] = {}
    eval_kwargs: Dict[str, Any] = {}
    test_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        train_kwargs["prefetch_factor"] = prefetch_factor
        eval_kwargs["prefetch_factor"] = prefetch_factor
        test_kwargs["prefetch_factor"] = prefetch_factor

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last_train,
            persistent_workers=persistent_workers,
            **train_kwargs,
        ),
        "eval": DataLoader(
            datasets["eval"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
            **eval_kwargs,
        ),
        **({"test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
            **test_kwargs,
        )} if include_test else {}),
    }
    return loaders
