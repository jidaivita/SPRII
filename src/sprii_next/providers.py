"""Only committed train/validation caches enter the development pipeline."""
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np
from .io import checked, development_path, read, sha, digest,plain

HORIZONS = (1, 2, 4, 8, 16)


@dataclass
class Batch:
    query: np.ndarray
    persistent: np.ndarray
    theta: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    horizon_index: np.ndarray
    target: np.ndarray
    rows: list


class SpringCache:
    environment = 'springworld'

    def __init__(self, descriptor):
        self.descriptor = descriptor
        for name in ('manifest', 'features_receipt', 'targets_receipt', 'checkpoint'):
            checked(descriptor[name], descriptor[name+'_sha256'])
        checked(descriptor['source_completion'],descriptor['source_completion_sha256'])
        completion=read(descriptor['source_completion'])
        self.synthetic_fixture=bool(descriptor.get('synthetic_fixture',False))
        if not self.synthetic_fixture:
            if completion['model_state_sha256']!=descriptor['model_state_sha256'] or completion['selected_step']!=10000:
                raise ValueError('source completion/cache identity differs')
            if descriptor['method']=='RelInfoNCE':
                if (completion['status']!='COMPLETE' or completion['source_seed']!=descriptor['source_seed']
                    or completion.get('smoke',False) or completion.get('recipe_id')!='fcrl_style_temporal_v1'):
                    raise ValueError('completed fixed-recipe contrastive source required')
            elif completion['name']!=('Split' if descriptor['method']=='Structure' else descriptor['method']):
                raise ValueError('source mechanism label differs')
        for path, expected in descriptor['native_files'].items():
            if Path(path).suffix!='.py' or sha(path)!=expected:raise ValueError('native code changed')
        sys.path[:0] = [str(development_path(p)) for p in descriptor['native_paths']]
        from persistbench.envs.visual_elastic_coupling.a_head_data import AHeadCasePlan
        from persistbench.envs.visual_elastic_coupling.a_head_features import AHeadFeatureCache
        from persistbench.envs.visual_elastic_coupling.a_head_targets import AHeadTargets
        if not self.synthetic_fixture:
            for name,module in list(sys.modules.items()):
                if name.startswith('persistbench.envs.visual_elastic_coupling') and getattr(module,'__file__',None):
                    filename=str(Path(module.__file__).resolve())
                    if filename not in descriptor['native_files']:raise ValueError('native import escaped committed source tree: '+filename)
        self.plan = AHeadCasePlan(read(descriptor['manifest']), seed=0, history_frames=96)
        feature_record = read(descriptor['features_receipt'])
        if feature_record['native_context_dim'] != 64:
            raise ValueError('this experiment requires source P64, not Native128 or reader memory')
        if feature_record['model_state_sha256'] != descriptor['model_state_sha256']:
            raise ValueError('cache and declared source state differ')
        self.features = AHeadFeatureCache(Path(descriptor['features_receipt']).parent, self.plan,
            receipt_sha256=descriptor['features_receipt_sha256'], model_state_sha256=descriptor['model_state_sha256'],
            bank_snapshot_sha256=feature_record['bank_snapshot_sha256'])
        self.labels = {s: AHeadTargets(Path(descriptor['targets_receipt']).parent, self.plan,
            receipt_sha256=descriptor['targets_receipt_sha256'], bank_snapshot_sha256=feature_record['bank_snapshot_sha256'],
            split=s, purpose='fit' if s=='train' else 'score') for s in ('train', 'validation')}
        self.normalization = plain(self.labels['train'].statistics.record)
        self.identity = digest(descriptor)
        # Fail before optimizing on missing support; never drop difficult rows.
        if not self.features._arrays['donor_support'].all() or not self.features._arrays['query_support'].all():
            raise ValueError('incomplete source cache support')
        self._assert_endpoint()

    def _assert_endpoint(self):
        from collections import Counter
        selected = [b for b in self.plan.base if b['split']=='validation' and b['kind']=='cold'
                    and b['stratum'] in ('continuous_new_systems', 'heldout_factorial_combinations')]
        counts = Counter(b['system_key'] for b in selected)
        strata = {b['system_key']:b['stratum'] for b in selected}
        if len(selected)!=600 or len(counts)!=100 or set(counts.values())!={6}:
            raise ValueError('SpringWorld cold-start endpoint must remain 600 cases / 100 systems')
        if Counter(strata.values()) != {'continuous_new_systems':64, 'heldout_factorial_combinations':36}:
            raise ValueError('SpringWorld cold-start strata changed')

    def donors(self, split):
        if split not in self.labels:
            raise PermissionError('development only')
        keys = [k for k in self.features._donor_index if self.plan.rows[k]['split']==split]
        indices = [self.features._donor_index[k] for k in keys]
        p = self.features._arrays['donor_slot'][indices, :64].copy()
        ids = np.asarray([self.plan.rows[k]['system_key'] for k in keys])
        theta = np.asarray([self.plan.systems[s]['theta'] for s in ids])
        return p, theta, ids, np.asarray(keys)

    def batch(self, cases, split):
        values = self.features.inputs(cases, arm='matched', device='cpu')
        labels = self.labels[split]
        rows=[]
        for c in cases:
            theta = self.plan.systems[c['system_key']]['theta']
            rows.append(dict(environment=self.environment, split=split, system_id=c['system_key'],
                query_id=c['case_id'], query_episode=c['query_episode'], query_anchor=c['anchor'], query_budget=c['q'],
                donor_id=c['matched_episode'], donor_episode=c['matched_episode'], donor_anchor=95,
                donor_system_id=c['system_key'], donor_condition='same_system_independent',
                horizon=c['horizon'], kind=c['kind'], stratum=c['stratum'], mass=theta[0], drag=theta[1], stiffness=theta[2],
                donor_mass=theta[0], donor_drag=theta[1], donor_stiffness=theta[2],
                primary=bool(c['kind']=='cold' and c['q']==0 and c['horizon']==16 and c['stratum'] in
                    ('continuous_new_systems','heldout_factorial_combinations')),
                split_weight=c['split_weight'], initial_state=None, contact_summary=None,
                S_mass=None,S_drag=None,S_stiffness=None,sensitivity_dominance=None))
        arrays = {k:v.numpy() for k,v in values.items()}
        for i,row in enumerate(rows): row['action_energy']=float(np.square(arrays['future_actions'][i]).sum())
        return Batch(arrays['query_embedding'], arrays['context_slot'][:,:64],
            np.asarray([self.plan.systems[c['system_key']]['theta'] for c in cases]), arrays['future_actions'],
            arrays['action_mask'], arrays['horizon_index'], labels.targets(cases), rows)

    def training_batch(self, seed, step, size):
        cases = self.plan.draw_training_batch(sampling_seed=seed, step=step, batch_size=size)
        return self.batch(cases, 'train')

    def evaluation_batches(self, size=256):
        pending=[]
        for i,b in enumerate(self.plan.base):
            if b['split']!='validation': continue
            for q in (0,1):
                for h in HORIZONS:
                    pending.append(self.plan.case(i,q=q,horizon=h,draw=0))
                    if len(pending)==size:
                        yield self.batch(pending,'validation');pending=[]
        if pending: yield self.batch(pending,'validation')


class PokeCache:
    environment = 'pokeworld'

    def __init__(self, descriptor):
        self.descriptor=descriptor
        checked(descriptor['checkpoint'],descriptor['checkpoint_sha256'])
        self.identity=digest(descriptor)
        self.data={}
        self.rows={}
        for split in ('train','validation'):
            path=checked(descriptor[split],descriptor[split+'_sha256'])
            with np.load(path,allow_pickle=False) as f:
                data={k:f[k] for k in f.files}
            import json
            meta=json.loads(str(data.pop('metadata_json')))
            if meta['split']!=split or meta['test_read'] is not False or meta['source_optimizer_steps']!=0:
                raise PermissionError('frozen development source required')
            if meta['source_sha256']!=descriptor['checkpoint_sha256'] or meta['condition']!=descriptor['method'] or meta['source_seed']!=descriptor['source_seed']:
                raise ValueError('Poke cache/source identity mismatch')
            self.rows[split]=json.loads(str(data.pop('rows_json')))
            if data['persistent'].shape!=(len(self.rows[split]),64):raise ValueError('P64 required')
            if any(r['system_id']!=r['donor_system_id'] or r['query_episode']==r['donor_episode'] for r in self.rows[split]):
                raise ValueError('same-system independent donor required')
            if any(not np.isfinite(a).all() for a in data.values()):raise ValueError('nonfinite cache')
            self.data[split]=data
            if split=='train':self.normalization=meta['normalization']
            elif meta['normalization']!=self.normalization:raise ValueError('normalization changed between splits')
            self.synthetic_fixture=meta.get('synthetic_fixture',False)
        if {r['system_id'] for r in self.rows['train']} & {r['system_id'] for r in self.rows['validation']}:
            raise ValueError('train/validation system overlap')
        train_theta={tuple(t) for t in self.data['train']['theta']}
        if train_theta & {tuple(t) for t in self.data['validation']['theta']}:
            raise ValueError('same physical tuple occurs in train and validation')

    def donors(self, split):
        if split not in self.data:raise PermissionError('development only')
        d=self.data[split];r=self.rows[split]
        # Use each actual donor window once, rather than weighting by reuse.
        _, ix=np.unique([x['donor_id'] for x in r],return_index=True)
        return d['persistent'][ix],d['theta'][ix],np.asarray([r[i]['system_id'] for i in ix]),np.asarray([r[i]['donor_id'] for i in ix])

    def batch(self, split, index, hi):
        d=self.data[split];index=np.asarray(index);hi=np.asarray(hi)
        h=np.asarray(HORIZONS)[hi]
        mask=(np.arange(16)[None,:]<h[:,None]).astype(np.float32)
        rows=[]
        for i,j in zip(index,hi):
            row=dict(self.rows[split][int(i)])
            row.update(horizon=HORIZONS[j],primary=HORIZONS[j]==16,query_id=row['base_id']+f':h{HORIZONS[j]}')
            for k,key in enumerate(('S_mass','S_drag','S_stiffness')):row[key]=float(d['sensitivity'][i,j,k])
            row['sensitivity_dominance']=float(d['dominance'][i,j])
            rows.append(row)
        return Batch(d['query'][index],d['persistent'][index],d['theta'][index],d['actions'][index]*mask[:,:,None],
            mask,hi.astype(np.int64),d['target'][index,hi],rows)

    def training_batch(self, seed, step, size):
        rng=np.random.default_rng(np.random.SeedSequence([seed,step]))
        index=rng.integers(len(self.rows['train']),size=size)
        hi=rng.integers(5,size=size)
        return self.batch('train',index,hi)

    def evaluation_batches(self,size=256):
        index=np.repeat(np.arange(len(self.rows['validation'])),5)
        hi=np.tile(np.arange(5),len(self.rows['validation']))
        for i in range(0,len(index),size):yield self.batch('validation',index[i:i+size],hi[i:i+size])


def provider(cfg, environment, method, source_seed):
    rows=[r for r in cfg['sources'] if (r['environment'],r['method'],r['source_seed'])==(environment,method,source_seed)]
    if len(rows)!=1:raise ValueError(f'exactly one source descriptor required: {environment}/{method}/{source_seed}')
    return (SpringCache if environment=='springworld' else PokeCache)(rows[0])
