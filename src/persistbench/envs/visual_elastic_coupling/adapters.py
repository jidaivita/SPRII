"""Strict public-input bridges; no private labels or simulator state cross here.

This module supplies wiring, not a trained state prediction head or an assay claim.
"""
from copy import deepcopy
import hashlib
import numpy as np
from persistbench.contracts import ExperienceBatch, QueryBatch, EpisodeContext

HISTORY_KEYS = frozenset({'observations', 'past_actions', 'past_action_mask', 'relative_times'})
QUERY_KEYS = HISTORY_KEYS | {'future_actions', 'horizon_seconds', 'target_spec'}

def _public(payload, allowed):
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f'non-public fields: {sorted(extra)}')
    return deepcopy(payload)

def a_history_arrays(payload):
    """24 strict raw frames -> 24 tokens and 23 executed actions (no boundary action)."""
    p = _public(payload, HISTORY_KEYS)
    obs = np.asarray(p['observations'], np.float32)
    actions = np.asarray(p['past_actions'], np.float32)
    mask = np.asarray(p['past_action_mask'], bool)
    if obs.ndim != 4 or obs.shape[1:] != (2,64,64) or len(obs) < 2:
        raise ValueError('A requires Lx2x64x64 visual tokens')
    if actions.shape != (len(obs),2) or mask.shape != (len(obs),) or mask[0] or not mask[1:].all():
        raise ValueError('strict action boundary/mask mismatch')
    if np.any(obs[0,1]) or np.any(actions[0]):
        raise ValueError('strict profile first difference/action must be zero')
    return obs[None].copy(), actions[None,1:].copy()

def a_predict(model, history, query):
    """Real pixel A layers; returns latent prediction, NOT a joint-state prediction.

    Caller controls train/eval and gradient context. Query is q=0 or q=1,
    with zero past actions. Long queries need a separately declared architecture.
    """
    import torch
    p = _public(query, QUERY_KEYS)
    if 'past_actions' in p or 'past_action_mask' in p:
        raise ValueError('A-query profile provides zero past actions')
    obs, act = a_history_arrays(history)
    qobs = np.asarray(p['observations'],np.float32)
    if qobs.ndim != 4 or qobs.shape[1:] != (2,64,64) or len(qobs) not in (1,2):
        raise ValueError('A query must have one or two raw frames')
    future = np.asarray(p['future_actions'],np.float32)
    h = len(future)
    if future.shape != (h,2) or h not in (1,4,16):
        raise ValueError('A predictor supports complete h=1,4,16 action intervals')
    device = next(model.parameters()).device
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    history_h = model.observation(tensor(obs))
    _, persistent, _ = model.codes(history_h,tensor(act))
    if persistent is None:
        raise ValueError('fresh-query donor bridge requires split model')
    query_h = model.observation(tensor(qobs[None]))[:,-1]
    transient = model.transient(query_h)
    padded = np.zeros((1,16,2),np.float32); padded[0,:h] = future
    mask = np.zeros((1,16),np.float32); mask[0,:h] = 1
    prediction = model.predictor(torch.cat([transient,persistent],-1),tensor(padded),tensor(mask),
                                 torch.tensor([(1,4,16).index(h)],device=device))
    return prediction

def z_experience(payload):
    p = _public(payload,HISTORY_KEYS)
    result = ExperienceBatch(system_keys=('opaque_case',), interaction_keys=('opaque_donor',),
        observations=p['observations'], actions=p['past_actions'], masks=p['past_action_mask'],
        timestamps=p['relative_times'], metadata={'observation_schema':'strict_image_difference_v1','action_schema':'previous_action_with_missing_first'})
    result.validate()
    return result

def z_query(payload):
    p = _public(payload,QUERY_KEYS)
    # Package allowed query fields together so optional past actions stay distinct.
    result = QueryBatch(query_keys=('opaque_query',), episode_tokens=('opaque_fresh',),
        observations={k:v for k,v in p.items() if k != 'future_actions'},
        actions=p['future_actions'], horizons=np.asarray([len(p['future_actions'])]),
        metadata={'observation_schema':'strict_query_packet_v1','action_schema':'future_executed_actions'})
    result.validate()
    return result

def _digest(value):
    if isinstance(value,np.ndarray):
        return hashlib.sha256(str((value.shape,str(value.dtype))).encode()+value.tobytes()).hexdigest()
    if isinstance(value,dict):
        return tuple((k,_digest(v)) for k,v in sorted(value.items()))
    return repr(value)

def run_fixed_artifact_conditions(agent, context, histories, query):
    """Evaluator-side condition names never enter method inputs.

    initialize must clear mutable memory while retaining the same trained artifact.
    Ingest donor, then reset fresh interaction while retaining permitted persistence.
    Each condition gets identical query bytes; no training occurs in this runner.
    """
    public_query = _public(query,QUERY_KEYS)
    fingerprint = _digest(public_query)
    results = {}
    artifact = getattr(agent, 'model', None)
    weights = None if artifact is None else _digest({k:v.detach().cpu().numpy().copy() for k,v in artifact.state_dict().items()})
    for condition, history in histories.items():
        agent.initialize(context)
        agent.reset(EpisodeContext('opaque_donor'))
        if history is not None:
            agent.ingest(z_experience(history))
        agent.reset(EpisodeContext('opaque_fresh'))
        packet = z_query(public_query)
        before = _digest(packet.observations),_digest(packet.actions)
        results[condition] = agent.respond(packet)
        if artifact is not None and weights != _digest({k:v.detach().cpu().numpy() for k,v in artifact.state_dict().items()}):
            raise RuntimeError('fixed artifact changed across conditions')
        if before != (_digest(packet.observations),_digest(packet.actions)) or _digest(public_query) != fingerprint:
            raise RuntimeError('method mutated fixed query')
    return results

# Preserve the import location while replacing raw-history retention. The direct
# a_predict path above remains an unchanged parity reference for the old math.
from .a_memory import PixelALatentAgent
