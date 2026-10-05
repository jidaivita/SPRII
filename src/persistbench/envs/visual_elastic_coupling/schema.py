"""Single-owner public/private boundary and strict causal window contract."""
from dataclasses import dataclass
import numpy as np

VERSION = "visual_elastic_coupling_v1_development"

@dataclass(frozen=True)
class Config:
    dt: float = 0.002
    control_dt: float = 0.05
    ell0: float = 0.35
    ell_min: float = 0.12
    radius: float = 0.05
    field_width: float = 2.0
    margin: float = 0.05
    force_max: float = 1.0
    resolution: int = 64

@dataclass(frozen=True)
class Parameters:
    m: float
    gamma: float
    k: float

    def validate(self):
        if not np.isfinite([self.m,self.gamma,self.k]).all() or self.m <= 0 or self.gamma < 0 or self.k <= 0:
            raise ValueError("invalid physical parameters")

@dataclass
class Episode:
    images: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    private_state: np.ndarray
    private_metadata: dict

    def validate(self):
        t = len(self.actions)
        if self.images.dtype != np.uint8 or self.images.ndim != 3 or len(self.images) != t+1:
            raise ValueError("expected T+1 uint8 grayscale frames")
        if self.actions.shape != (t,2) or self.private_state.shape != (t+1,8) or self.timestamps.shape != (t+1,):
            raise ValueError("episode indexing/shape mismatch")
        if not np.isfinite(self.actions).all() or np.any(np.linalg.norm(self.actions,axis=1)>1+1e-7):
            raise ValueError("invalid normalized action")

def history_payload(episode, start, stop):
    """Inclusive raw-frame bounds; identity and simulator labels never projected."""
    episode.validate()
    if not 0 <= start <= stop < len(episode.images):
        raise ValueError("history bounds")
    frames = episode.images[start:stop+1].copy()
    x = frames.astype(np.float32)/255
    difference = np.zeros_like(x)
    difference[1:] = x[1:] - x[:-1]
    actions = np.zeros((len(frames),2),np.float32)
    actions[1:] = episode.actions[start:stop]
    mask = np.ones(len(frames),bool); mask[0] = False
    return dict(observations=np.stack([x,difference],axis=1), past_actions=actions,
                past_action_mask=mask, relative_times=episode.timestamps[start:stop+1]-episode.timestamps[start])

def query_packet(episode, anchor, q, horizon, provide_past_actions=False):
    if q < 0 or horizon < 1 or anchor-q < 0 or anchor+horizon >= len(episode.images):
        raise ValueError("incomplete query or future interval")
    result = history_payload(episode,anchor-q,anchor)
    if not provide_past_actions:
        result.pop("past_actions"); result.pop("past_action_mask")
    result["future_actions"] = episode.actions[anchor:anchor+horizon].copy()
    result["horizon_seconds"] = float(episode.timestamps[anchor+horizon]-episode.timestamps[anchor])
    result["target_spec"] = "joint_state_delta_8d"
    return result
