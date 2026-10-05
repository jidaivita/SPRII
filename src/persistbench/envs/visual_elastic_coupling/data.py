"""Independent episodes and evaluator-owned condition envelopes."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .schema import Config, Episode, history_payload, query_packet

def action_library(rng, transitions, kind):
    a = np.zeros((transitions,2),np.float32)
    if kind in ("static","glide","free"):
        return a
    if kind != "forced":
        raise ValueError("unknown action template")
    # Draws are independent of theta; balanced pulse pairs limit global drift.
    cursor = 0
    while cursor < transitions:
        angle = rng.uniform(0,2*np.pi)
        amplitude = rng.choice([.25,.5,.75])
        length = int(rng.choice([2,4,8]))
        pulse = amplitude*np.array([np.cos(angle),np.sin(angle)])
        a[cursor:min(cursor+length,transitions)] = pulse
        cursor += length
        a[cursor:min(cursor+length,transitions)] = -pulse
        cursor += length + int(rng.choice([8,16,24]))
    return a

def initial_state(rng,kind,config):
    center = rng.uniform(-.08,.08,2)
    angle = rng.uniform(0,2*np.pi)
    n = np.array([np.cos(angle),np.sin(angle)])
    length = config.ell0 + (rng.choice([-.04,.04]) if kind in ("free","forced") else 0)
    v = rng.uniform(.1,.2)*np.array([np.cos(angle+.5),np.sin(angle+.5)]) if kind == "glide" else np.zeros(2)
    return np.concatenate([center-length*n/2,center+length*n/2,v,v])

def generate_episode(parameters,seed,kind="forced",transitions=96,config=Config()):
    from .physics import trajectory
    from .rendering import render_states
    children = np.random.SeedSequence(seed).spawn(3)
    state = initial_state(np.random.default_rng(children[0]),kind,config)
    actions = action_library(np.random.default_rng(children[1]),transitions,kind)
    states = trajectory(parameters,state,actions,config)
    ep = Episode(render_states(states,config), actions,
                 np.arange(len(states),dtype=np.float64)*config.control_dt,states,
                 dict(theta=asdict(parameters),seed=seed,kind=kind,config=asdict(config),
                      episode_key=f"development_{seed}_{kind}",split="development",
                      appearance_policy="fixed_parameter_independent",status="valid"))
    ep.validate()
    return ep

def save_episode(episode,path):
    """Store visible and private assets separately; NPZ filenames are never model inputs."""
    path = Path(path); path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/"visible.npz",images=episode.images,actions=episode.actions,timestamps=episode.timestamps)
    np.savez_compressed(path/"private.npz",state=episode.private_state)
    (path/"private.json").write_text(json.dumps(episode.private_metadata,indent=2)+"\n")

def digest_packet(packet):
    h = hashlib.sha256()
    for key in sorted(packet):
        h.update(key.encode())
        value = packet[key]
        if isinstance(value,np.ndarray):
            h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
        else:
            h.update(json.dumps(value,sort_keys=True).encode())
    return h.hexdigest()

def fixed_query_cases(query,donors,anchor=32,q=1,horizon=16,frames=24):
    packet = query_packet(query,anchor,q,horizon,False)
    target = query.private_state[anchor+horizon]-query.private_state[anchor]
    digest = digest_packet(packet)
    cases = []
    for condition,donor in donors.items():
        if donor is not None and donor.private_metadata["episode_key"] == query.private_metadata["episode_key"]:
            raise ValueError("donor must be an independent episode")
        payload = None if donor is None else history_payload(donor,0,frames-1)
        cases.append(dict(query=packet,history=payload,private=dict(condition=condition,
            target=target.tolist(),query_hash=digest,target_hash=hashlib.sha256(target.tobytes()).hexdigest(),
            query_episode=query.private_metadata["episode_key"],
            donor_episode=None if donor is None else donor.private_metadata["episode_key"],
            protocol="fixed_artifact_history_intervention",split="development",
            budget=dict(query_frames=q+1,observed_query_transitions=q,provided_query_past_actions=0,
                        future_actions=horizon,donor_frames=0 if donor is None else frames,
                        donor_transitions=0 if donor is None else frames-1))))
    return cases

def relation_edges(records,relation):
    """Private sampler; explicit shared-level bank required, no cross-split edges."""
    keys = {"G1":("m",),"G2":("m","gamma"),"G3":("m","gamma","k")}[relation]
    changed = {"G1":("gamma","k"),"G2":("k",),"G3":()}[relation]
    return [(i,j) for i,a in enumerate(records) for j,b in enumerate(records)
            if i != j and a["split"] == b["split"] and a["episode_key"] != b["episode_key"]
            and all(a["theta"][key] == b["theta"][key] for key in keys)
            and all(a["theta"][key] != b["theta"][key] for key in changed)]
