"""Private A head case plans and content-bound, role-specific data access.

No training occurs here. Train/validation cases are declared without inspecting
labels or filtering failed trajectories. Query, donor and label access remain
separate; a missing donor does not erase an otherwise available Null query.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np
from .a_head_statistics import HORIZONS
from .a_pairing import digest
from .dataset_snapshot import checked_asset,stable_digest,content_digest
from .schema import Episode,history_payload,query_packet

SCHEMA='vec.A-head-case-plan.v1'
SELECTION_STRATA=('continuous_new_systems','heldout_factorial_combinations')


def _immutable(value):
    if isinstance(value,dict):return MappingProxyType({k:_immutable(v) for k,v in value.items()})
    if isinstance(value,(list,tuple)):return tuple(_immutable(v) for v in value)
    return value


class AHeadCasePlan:
    def __init__(self,manifest,*,seed,history_frames=24):
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('registered uint32 head sampling seed required')
        if type(history_frames) is not int or history_frames not in (24,48,96):raise ValueError('unregistered A donor frame budget')
        rows=manifest['episodes']
        if any(r['split'] not in ('train','validation') for r in rows):raise PermissionError('test episodes forbidden in A head fitting bank')
        self.seed=seed;self.frames=history_frames;self.manifest_semantic_sha256=digest(manifest)
        self.rows={};self.systems={};self.groups={};self.pools={};seen_episodes=set()
        for original in rows:
            row=copy.deepcopy(original);key=row['episode_key'];system=row['system_key']
            if key in seen_episodes:raise ValueError('duplicate A head source episode')
            seen_episodes.add(key)
            theta=row['theta']
            if len(theta)!=3 or not np.isfinite(theta).all() or min(theta)<=0:raise ValueError('invalid system parameters')
            entry=dict(split=row['split'],stratum=row['stratum'],theta=theta)
            if system in self.systems and self.systems[system]!=entry:raise ValueError('physical system changes split,stratum or parameters')
            self.systems[system]=entry;self.pools.setdefault(system,{})
            if row['kind'] not in ('forced','cold','moving'):continue
            self.rows[key]=row;self.pools[system].setdefault(row['kind'],[]).append(key)
        if len({tuple(x['theta']) for x in self.systems.values()})!=len(self.systems):raise ValueError('physical tuple repeated across declared systems')
        for system,entry in self.systems.items():self.groups.setdefault((entry['split'],entry['stratum']),[]).append(system)
        if not any(x['split']=='train' for x in self.systems.values()) or not any(x['split']=='validation' and x['stratum'] in SELECTION_STRATA for x in self.systems.values()):
            raise ValueError('complete train and declared validation-selection populations required')
        for group,systems in self.groups.items():
            systems.sort()
            if len(systems)<2:raise ValueError('at least two systems per wrong-donor population required')
            signatures=[]
            for system in systems:
                if set(self.pools[system])!={'forced','cold','moving'}:raise ValueError('missing declared A query or donor episode kind')
                counts=[]
                for kind in ('forced','cold','moving'):
                    ordered=sorted(self.pools[system][kind],key=lambda k:self.rows[k]['replicate'])
                    if [self.rows[k]['replicate'] for k in ordered]!=list(range(len(ordered))):raise ValueError('episode replicate positions incomplete')
                    self.pools[system][kind]=ordered;counts.append(len(ordered))
                    for key in ordered:
                        row=self.rows[key]
                        if type(row['requested_frames']) is not int or type(row['raw_frames']) is not int or not 1<=row['raw_frames']<=row['requested_frames']:
                            raise ValueError('invalid planned/actual episode frame count')
                        if kind=='forced' and row['requested_frames']<history_frames:raise ValueError('requested donor is too short')
                        if kind!='forced' and (type(row['anchor']) is not int or row['anchor']<1 or row['anchor']+16>=row['requested_frames']):
                            raise ValueError('requested query cannot support common q0/1 and h16')
                signatures.append(tuple(counts))
            if len(set(signatures))!=1:raise ValueError('episode counts differ within a balanced donor population')
        self.base=[]
        split_counts={s:sum(x['split']==s for x in self.systems.values()) for s in ('train','validation')}
        selection_count=sum(x['split']=='validation' and x['stratum'] in SELECTION_STRATA for x in self.systems.values())
        for system in sorted(self.systems):
            entry=self.systems[system]
            for kind in ('cold','moving'):
                pool=self.pools[system][kind]
                for key in pool:
                    row=self.rows[key]
                    self.base.append(dict(query_episode=key,system_key=system,split=entry['split'],stratum=entry['stratum'],kind=kind,
                        replicate=row['replicate'],anchor=row['anchor'],split_probability=1./(split_counts[entry['split']]*2*len(pool)),
                        selection_probability=1./(selection_count*2*len(pool)) if entry['split']=='validation' and entry['stratum'] in SELECTION_STRATA else 0.))
        self._settings=self._state_digest()
        self.plan_sha256=digest(dict(schema=SCHEMA,manifest_semantic_sha256=self.manifest_semantic_sha256,state_sha256=self._settings,
            seed=seed,history_frames=history_frames,query_budgets=[0,1],horizons=list(HORIZONS),selection_strata=list(SELECTION_STRATA)))
        # Frozen, detached containers make per-case validation independent of
        # total bank size. Rehashing all metadata per sampled case would turn
        # batch construction into O(batch_size * bank_size).
        for name in ('rows','systems','groups','pools','base'):setattr(self,name,_immutable(getattr(self,name)))
        self._identity=self._identity_stamp()

    def _state_digest(self):
        return digest([self.seed,self.frames,self.rows,self.systems,[[list(k),v] for k,v in sorted(self.groups.items())],self.pools,self.base])
    def _identity_stamp(self):
        return (self.seed,self.frames,self.manifest_semantic_sha256,self.plan_sha256,
                *(id(getattr(self,name)) for name in ('rows','systems','groups','pools','base')))
    def _guard(self):
        if self._identity_stamp()!=self._identity:raise ValueError('A head case plan metadata changed')

    def _rng(self,draw,group,kind,replicate):
        entropy=int(digest([self.seed,draw,list(group),kind,replicate])[:16],16)
        return np.random.default_rng(entropy)

    def case(self,index,*,q,horizon,draw=0):
        self._guard()
        if type(index) is not int or not 0<=index<len(self.base) or type(q) is not int or q not in (0,1) or type(horizon) is not int or horizon not in HORIZONS:
            raise ValueError('invalid A head case/query/horizon')
        if type(draw) is not int or not 0<=draw<2**32:raise ValueError('invalid registered donor draw')
        base=self.base[index];group=(base['split'],base['stratum']);systems=self.groups[group]
        rng=self._rng(draw,group,base['kind'],base['replicate']);offset=int(rng.integers(1,len(systems)))
        wrong=systems[(systems.index(base['system_key'])+offset)%len(systems)]
        donors=self.pools[base['system_key']]['forced'];rank=int(rng.integers(len(donors)))
        matched=donors[rank];mismatched=self.pools[wrong]['forced'][rank]
        row=self.rows[base['query_episode']]
        result=dict(base,index=index,q=q,horizon=horizon,donor_draw=draw,history_frames=self.frames,
            matched_episode=matched,wrong_episode=mismatched,matched_start=0,wrong_start=0,
            wrong_system=wrong,donor_episode_probability=1./len(donors),wrong_system_probability=1./(len(systems)-1),
            matched_observed_support=self.rows[matched]['raw_frames']>=self.frames,
            wrong_observed_support=self.rows[mismatched]['raw_frames']>=self.frames,
            query_observed_support=row['raw_frames']>base['anchor'],future_target_support=row['raw_frames']>base['anchor']+horizon,
            selection_weight=base['selection_probability']/10,split_weight=base['split_probability']/10,plan_sha256=self.plan_sha256)
        result['case_id']=digest([self.plan_sha256,index,q,horizon,draw]);result['case_sha256']=digest(result)
        return result

    def validate_case(self,case):
        expected=self.case(case['index'],q=case['q'],horizon=case['horizon'],draw=case['donor_draw'])
        if digest(case)!=digest(expected):raise ValueError('A head case differs from planned common query/donors')

    def draw_training_batch(self,*,sampling_seed,step,batch_size):
        self._guard()
        if any(type(x) is not int or x<0 or x>=2**32 for x in (sampling_seed,step)) or type(batch_size) is not int or batch_size<1:
            raise ValueError('invalid head sampling seed/step/batch')
        indices=[i for i,b in enumerate(self.base) if b['split']=='train'];weights=np.array([self.base[i]['split_probability'] for i in indices])
        if not np.isclose(weights.sum(),1.):raise ValueError('training distribution lost complete denominator')
        rng=np.random.default_rng(np.random.SeedSequence([sampling_seed,step]))
        chosen=rng.choice(indices,batch_size,p=weights);qs=rng.integers(0,2,batch_size);hs=rng.choice(HORIZONS,batch_size)
        return [self.case(int(i),q=int(q),horizon=int(h),draw=step) for i,q,h in zip(chosen,qs,hs)]

    def inventory(self):
        self._guard()
        return dict(schema=SCHEMA,status='CANDIDATE_NOT_FROZEN',plan_sha256=self.plan_sha256,
            manifest_semantic_sha256=self.manifest_semantic_sha256,seed=self.seed,history_frames=self.frames,
            base_queries=len(self.base),expanded_cases_per_donor_draw=len(self.base)*10,
            systems={s:sum(x['split']==s for x in self.systems.values()) for s in ('train','validation')},
            selection_strata=list(SELECTION_STRATA),query_budgets=[0,1],horizons=list(HORIZONS),
            training_weight='uniform physical system,query kind,episode within kind,query budget,horizon',
            donor_rule='same-split/stratum balanced wrong-system cyclic permutation per query kind,replicate and donor draw; uniform episode rank',
            normalization_unit='unique training query episode/horizon and unique physical training system',
            failure_policy='planned identities retained; support flags explicit; no donor/query rejection resampling',
            formal_training=False,test_read=False)


class AHeadDataAccess:
    """Training/validation source reader, not a sealed-test authorization object."""
    def __init__(self,root,plan,*,snapshot_sha256):
        self.root=Path(root);self.plan=plan;self.audit=[]
        snapshot_path=checked_asset(root,'BANK_SNAPSHOT.json')
        if stable_digest(snapshot_path)['sha256']!=snapshot_sha256:raise ValueError('A head snapshot does not match supplied commitment')
        self.snapshot=json.loads(snapshot_path.read_text())
        if self.snapshot.get('schema')!='vec.bank-content-snapshot.v1.1' or self.snapshot.get('status')!='PASS' or self.snapshot.get('test_read') or content_digest(self.snapshot['files'])!=self.snapshot['content_sha256']:
            raise ValueError('unqualified A head data snapshot')
        self.files={r['path']:r for r in self.snapshot['files']}
        if len(self.files)!=len(self.snapshot['files']):raise ValueError('duplicate data snapshot path')
        self.snapshot_sha256=snapshot_sha256
        for name in ('BANK_CONFIG.json','BANK_REPORT.json','MANIFEST.private.json'):
            self._bound(name)
        config=json.loads((self.root/'BANK_CONFIG.json').read_text());report=json.loads((self.root/'BANK_REPORT.json').read_text())
        if any(x.get('test_read') or x.get('test_generated') for x in (config,report)) or report.get('execution_errors'):
            raise PermissionError('test or failed execution bank forbidden for A head fitting')
        manifest=json.loads((self.root/'MANIFEST.private.json').read_text())
        if digest(manifest)!=plan.manifest_semantic_sha256:raise ValueError('A head plan uses another bank manifest')
        plan._guard()

    def _bound(self,relative):
        if relative not in self.files:raise ValueError('asset absent from committed bank snapshot')
        path=checked_asset(self.root,relative);expected=self.files[relative]
        if stable_digest(path)!={k:expected[k] for k in ('bytes','sha256')}:raise ValueError('A head source content changed')
        return path

    def _load(self,row,*,private,purpose):
        if row['split'] not in ('train','validation'):raise PermissionError('test data forbidden in A head fitting reader')
        if private and (purpose not in ('fit','score') or (purpose=='fit' and row['split']!='train')):
            raise PermissionError('nontraining label access forbidden for A head fitting')
        relative=row['private_state_path'] if private else row['assets']['128']['path']
        path=self._bound(relative)
        with np.load(path,allow_pickle=False) as data:result={k:data[k].copy() for k in data.files}
        self._bound(relative)
        self.audit.append(dict(episode_key=row['episode_key'],split=row['split'],kind='private_label' if private else 'public_observation',purpose=purpose,path=relative))
        if private:
            if set(result)!={'state'} or result['state'].shape!=(row['raw_frames'],8) or not np.isfinite(result['state']).all():raise ValueError('invalid A physical training/validation labels')
            return result['state']
        if set(result)!={'images','actions','timestamps'}:raise ValueError('unexpected public A head source fields')
        n=row['raw_frames'];images=result['images'];actions=result['actions'];times=result['timestamps']
        if images.dtype!=np.uint8 or images.shape!=(n,128,128) or actions.shape!=(n-1,2) or times.shape!=(n,) or not np.isfinite(actions).all() or np.any(np.linalg.norm(actions,axis=1)>1+1e-7) or not np.allclose(times,np.arange(n)*.05,atol=1e-8,rtol=0):
            raise ValueError('invalid public A head image/action/time support')
        return Episode(images,actions,times,np.zeros((n,8)),{})

    def query(self,case):
        from .adapters import z_query
        self.plan.validate_case(case)
        if not case['query_observed_support'] or not case['future_target_support']:raise ValueError('planned A query/action/target support missing; no replacement')
        episode=self._load(self.plan.rows[case['query_episode']],private=False,purpose='query')
        return z_query(query_packet(episode,case['anchor'],case['q'],case['horizon']))

    def public_episode(self,episode_key,*,purpose):
        """Public-only extraction path: read a unique source once for many views."""
        self.plan._guard()
        if episode_key not in self.plan.rows:raise ValueError('episode absent from A head plan')
        row=self.plan.rows[episode_key]
        if purpose not in ('query','donor') or (purpose=='query' and row['kind'] not in ('cold','moving')) or (purpose=='donor' and row['kind']!='forced'):
            raise ValueError('episode role differs from the A head extraction plan')
        return self._load(row,private=False,purpose=purpose)

    def history(self,case,*,source):
        from .adapters import z_experience
        self.plan.validate_case(case)
        if source not in ('matched','wrong'):raise ValueError('unregistered donor source')
        if not case[source+'_observed_support']:raise ValueError('planned A donor support missing; no replacement')
        episode=self._load(self.plan.rows[case[source+'_episode']],private=False,purpose='donor')
        start=case[source+'_start']
        return z_experience(history_payload(episode,start,start+case['history_frames']-1))

    def target(self,case,*,purpose):
        self.plan.validate_case(case)
        row=self.plan.rows[case['query_episode']]
        if purpose not in ('fit','score') or (purpose=='fit' and row['split']!='train'):raise PermissionError('validation labels cannot enter A head fitting')
        if not case['future_target_support']:raise ValueError('planned A physical target missing; no replacement')
        state=self._load(row,private=True,purpose=purpose);anchor=case['anchor']
        return (state[anchor+case['horizon']]-state[anchor]).copy()

    def target_row(self,index,*,purpose):
        """All five targets from one permitted label read; missing slots explicit."""
        self.plan._guard()
        if type(index) is not int or not 0<=index<len(self.plan.base):raise ValueError('invalid A target row')
        base=self.plan.base[index];row=self.plan.rows[base['query_episode']]
        if purpose not in ('fit','score') or (purpose=='fit' and row['split']!='train'):
            raise PermissionError('nontraining target row cannot enter A head fitting')
        support=np.asarray([row['raw_frames']>base['anchor']+h for h in HORIZONS],bool)
        targets=np.zeros((len(HORIZONS),8),np.float64)
        if support.any():
            state=self._load(row,private=True,purpose=purpose)
            with np.errstate(over='ignore',invalid='ignore'):
                for hi,h in enumerate(HORIZONS):
                    if support[hi]:targets[hi]=state[base['anchor']+h]-state[base['anchor']]
            if not np.isfinite(targets).all():raise ValueError('nonfinite A physical target displacement')
        return targets,support

    def known_parameters(self,case,*,purpose):
        self.plan.validate_case(case)
        if purpose not in ('fit','score') or (purpose=='fit' and case['split']!='train'):raise PermissionError('validation parameters cannot enter A Oracle fitting statistics')
        self.audit.append(dict(system_key=case['system_key'],split=case['split'],kind='Oracle_parameters',purpose=purpose))
        return np.asarray(self.plan.systems[case['system_key']]['theta'],float).copy()

    def verify_all(self,workers=16):
        from .dataset_snapshot import verify
        if stable_digest(checked_asset(self.root,'BANK_SNAPSHOT.json'))['sha256']!=self.snapshot_sha256:raise ValueError('bank snapshot changed')
        return verify(self.root,self.snapshot,workers)
