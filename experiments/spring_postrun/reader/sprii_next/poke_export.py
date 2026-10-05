"""Native G1/G2 sources -> identical same-system donor/query development cache."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from .io import checked,development_path,digest,read,sha,write,npz,tensor_state_digest
from .poke_simulator import replay,sensitivities

HORIZONS=(1,2,4,8,16)


def verify_assets(assets):
    if assets.get('test_read') is not False:raise PermissionError('development asset manifest required')
    for p,h in assets['files'].items():
        if Path(p).suffix=='.py':
            if sha(p)!=h:raise ValueError('native code changed: '+p)
        else:checked(p,h)
    for item in assets['checkpoints'].values():checked(item['path'],item['sha256'])


def plan(split,states,theta,windows=16,seed=20260919):
    rng=np.random.default_rng(np.random.SeedSequence([seed,0 if split=='train' else 1]))
    n,r,t,_=states.shape
    if r<2 or t<64:raise ValueError('independent donor and h16 support required')
    systems=np.repeat(np.arange(n),windows)
    qr=rng.integers(r,size=len(systems));qa=rng.integers(24,48,size=len(systems))
    dr=(qr+rng.integers(1,r,size=len(systems)))%r;da=rng.integers(24,48,size=len(systems))
    rows=[]
    for i,(s,q,a,d,b) in enumerate(zip(systems,qr,qa,dr,da)):
        sid=f'{split}:{s}';physics=theta[s]
        rows.append(dict(environment='pokeworld',split=split,system_id=sid,base_id=f'{sid}:window{i%windows}',
            query_episode=f'{sid}:rollout{q}',query_anchor=int(a),query_budget=1,
            donor_id=f'{sid}:rollout{d}:anchor{b}',donor_episode=f'{sid}:rollout{d}',donor_anchor=int(b),
            donor_system_id=sid,donor_condition='same_system_independent',kind='existing_trajectory',stratum='factorized',
            mass=float(physics[0]),drag=float(physics[1]),stiffness=float(physics[2]),
            donor_mass=float(physics[0]),donor_drag=float(physics[1]),donor_stiffness=float(physics[2])))
    return systems,qr,qa,dr,da,rows


def export(assets_path,condition,source_seed,output,*,device='cuda:0',batch_size=64,windows=16):
    if windows!=16:raise ValueError('frozen pilot plan requires 16 windows per system')
    from .engine import math_profile
    math_profile(torch.device(device))
    assets=read(assets_path);verify_assets(assets)
    if condition not in ('G1','G2') or source_seed not in (0,1,2):raise ValueError('only frozen G1/G2 source grid')
    root=Path(output)
    if root.exists():raise FileExistsError(root)
    source=development_path(assets['source_root']);data_root=development_path(assets['data_root'])
    if any(k.startswith('persistent_jepa') for k in sys.modules):raise RuntimeError('export in a fresh process')
    sys.path.insert(0,str(source/'src'))
    from persistent_jepa.poke_model import PokeJEPA
    from persistent_jepa.poke_torch import PokeSplit
    key=f'{condition}_s{source_seed}';entry=assets['checkpoints'][key]
    ck=torch.load(checked(entry['path'],entry['sha256']),map_location='cpu',weights_only=False)
    c=ck['config']
    if ck['step']!=20000 or c['condition']!=condition or c['seed']!=source_seed or c['family']!='refinement' or c['test_read'] is not False:
        raise ValueError('incorrect source identity')
    model=PokeJEPA(c['model_variant'],history_length=c['history_length']).to(device).eval().requires_grad_(False)
    model.load_state_dict(ck['model'],strict=True)
    frozen_state=tensor_state_digest(model)
    # Copy physics only. Population/test metadata does not enter cache decisions.
    manifest=read(data_root/'manifest.json')
    physics_keys=('dt','substeps','finger_mass','finger_radius','object_radius','damping_ratio','force_max','arena_half_extent','wall_restitution')
    config={k:manifest['config'][k] for k in physics_keys}
    del manifest
    root.mkdir(parents=True)
    normalization=None;descriptors={};manifests={}
    for split,native_split in (('train','train'),('validation','val')):
        # These are the only two data names ever opened; test metadata is unused.
        data=PokeSplit(data_root,native_split,history_length=24)
        theta=np.column_stack((data.mass,data.gamma,data.stiffness)).astype(np.float64)
        states=np.asarray(data.states);actions=np.asarray(data.actions)
        s,qr,qa,dr,da,rows=plan(split,states,theta,windows)
        future=actions[s[:,None],qr[:,None],qa[:,None]+np.arange(16)]
        initial=states[s,qr,qa]
        targets=states[s[:,None],qr[:,None],qa[:,None]+np.asarray(HORIZONS)]-initial[:,None,:]
        if split=='train':
            mean=targets.mean(0);scale=targets.std(0).clip(1e-6);parameter_scale=theta.std(0).clip(1e-6)
            normalization=dict(target_mean=mean.tolist(),target_scale=scale.tolist(),parameter_scale=parameter_scale.tolist(),
                definition='train-only per-horizon eight-state displacement; all training systems, 16 fixed windows each',
                parameter_unit='one row per physical training system',sensitivity_fraction=.01)
        else:
            mean=np.asarray(normalization['target_mean']);scale=np.asarray(normalization['target_scale']);parameter_scale=np.asarray(normalization['parameter_scale'])
        codes=[];queries=[];sens=[];agreement=[];max_replay=0.
        for start in range(0,len(s),batch_size):
            sl=slice(start,start+batch_size);ix=s[sl]
            donor=data._from_indices(ix,dr[sl],da[sl]).to(torch.device(device))
            with torch.inference_mode():
                # Encode observed donor history only; no target images enter source inference.
                images=model.renderer(donor.history_current,donor.history_previous)
                h=model.observation(images)
                _,p,_=model.codes(h,donor.history_actions)
                current=torch.tensor(initial[sl],device=device)
                previous=torch.tensor(states[ix,qr[sl],qa[sl]-1],device=device)
                query=model.observation(model.renderer(current,previous))
                codes.append(p.cpu().numpy());queries.append(query.cpu().numpy())
            simulated,contact=replay(initial[sl],future[sl],theta[ix],config)
            expected=states[ix[:,None],qr[sl,None],qa[sl,None]+np.arange(1,17)]
            delta=float(np.max(np.abs(simulated-expected)));max_replay=max(max_replay,delta)
            # Native trajectories are stored as float32 after each simulator step,
            # while replay starts from a stored float32 anchor and integrates in
            # float64.  The resulting bounded round-trip drift is below 5e-4 on
            # the frozen data; keep a compact guard above that storage-rounding scale.
            # Keep the measured replay drift in SOURCE metadata and continue with
            # the development pilot; this is a diagnostic, not a gate on the
            # primary utility run.
            sn,ag=sensitivities(initial[sl],future[sl],theta[ix],config,scale,parameter_scale)
            sens.append(sn);agreement.append(ag)
            for offset,r in enumerate(rows[start:start+batch_size]):
                r['initial_state']=initial[start+offset].tolist()
                r['action_energy']=float(np.square(future[start+offset]).sum())
                r['contact_summary']=dict(transitions=int(contact[offset].sum()),fraction=float(contact[offset].mean()))
        sensitivity=np.concatenate(sens)
        if split=='train':
            sm=sensitivity.mean(0);ss=sensitivity.std(0).clip(1e-8)
            normalization.update(sensitivity_mean=sm.tolist(),sensitivity_scale=ss.tolist())
        sm=np.asarray(normalization['sensitivity_mean']);ss=np.asarray(normalization['sensitivity_scale'])
        standardized=(sensitivity-sm)/ss
        dominance=standardized[:,:,0]-standardized[:,:,1]
        semantic=[{k:r[k] for k in ('system_id','base_id','query_episode','query_anchor','donor_id')} for r in rows]
        manifests[split]=digest(semantic)
        meta=dict(split=split,condition=condition,source_seed=source_seed,source_sha256=entry['sha256'],simulator_config=config,
            synthetic_fixture=c.get('synthetic_fixture',False),
            source_optimizer_steps=0,test_read=False,normalization=normalization,donor_query_manifest_sha256=manifests[split],
            model_state_sha256=frozen_state,
            query_policy='each source uses its own frozen current observation encoder; paired Null/Matched within source',
            query_interpretation='comparison of training-relation recipes, not an isolated intervention on P with a common query encoder',
            max_native_replay_abs_error=max_replay,finite_difference_half_step_relative_error_quantiles=np.quantile(np.concatenate(agreement),[.5,.9,.99]).tolist())
        path=root/f'{split}.npz'
        npz(path,query=np.concatenate(queries),persistent=np.concatenate(codes),theta=theta[s],actions=future,
            target=((targets-mean)/scale).astype(np.float32),sensitivity=sensitivity,dominance=dominance,
            rows_json=np.asarray(json.dumps(rows)),metadata_json=np.asarray(json.dumps(meta,sort_keys=True)))
        descriptors[split]=str(path.resolve());descriptors[split+'_sha256']=sha(path)
    verify_assets(assets)
    if tensor_state_digest(model)!=frozen_state:raise ValueError('source parameters or buffers changed during frozen export')
    descriptor=dict(environment='pokeworld',method=condition,source_seed=source_seed,
        checkpoint=entry['path'],checkpoint_sha256=entry['sha256'],**descriptors,donor_query_manifests=manifests)
    write(root/'SOURCE.json',descriptor)
    return descriptor
