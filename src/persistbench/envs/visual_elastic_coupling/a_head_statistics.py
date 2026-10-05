"""A fresh-head normalization, fitted once on explicit training cases/systems."""
import copy
import hashlib
import json
import numpy as np

HORIZONS = (1, 2, 4, 8, 16)
FLOORS = np.array([1e-5]*4+[1e-4]*4)
SCHEMA = 'vec.A-fresh-head-statistics.v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def parameter_coordinates(theta):
    x=np.asarray(theta,np.float64)
    if x.shape[-1:]!=(3,) or not np.isfinite(x).all() or np.any(x[...,0]<=0) or np.any(x[...,1]<0) or np.any(x[...,2]<=0):
        raise ValueError('invalid known physical parameters')
    return np.stack((np.log(x[...,0]),x[...,1],np.log(x[...,2])),axis=-1)


class _Moments:
    def __init__(self,dim):self.n=0;self.mean=np.zeros(dim);self.m2=np.zeros(dim)
    def add(self,x):
        self.n+=1;d=x-self.mean;self.mean+=d/self.n;self.m2+=d*(x-self.mean)
    def export(self,floor):
        if self.n<2:raise ValueError('at least two training observations per normalization group required')
        return dict(count=self.n,mean=self.mean.tolist(),scale=np.maximum(np.sqrt(np.maximum(self.m2,0)/self.n),floor).tolist())


class FitAHeadStatistics:
    def __init__(self):
        self.targets={h:_Moments(8) for h in HORIZONS};self.oracle=_Moments(3)
        self.seen_targets=set();self.seen_systems=set();self.seen_tuples=set()
        self.target_digest=hashlib.sha256();self.oracle_digest=hashlib.sha256()

    def add_target(self,*,split,case_key,horizon,delta):
        # Check permissions before converting/inspecting any label value.
        if split!='train':raise PermissionError('A head target statistics accept training labels only')
        if type(horizon) is not int or horizon not in HORIZONS or not isinstance(case_key,str) or not case_key:
            raise ValueError('invalid registered training target identity/horizon')
        key=(case_key,horizon)
        if key in self.seen_targets:raise ValueError('duplicate training target')
        x=np.asarray(delta,np.float64)
        if x.shape!=(8,) or not np.isfinite(x).all():raise ValueError('invalid physical state displacement')
        self.targets[horizon].add(x);self.seen_targets.add(key)
        self.target_digest.update(json.dumps(key).encode()+x.astype('<f8').tobytes())

    def add_system(self,*,split,system_key,theta):
        if split!='train':raise PermissionError('Oracle normalization accepts training systems only')
        if not isinstance(system_key,str) or not system_key or system_key in self.seen_systems:
            raise ValueError('duplicate or invalid Oracle training system')
        x=parameter_coordinates(theta)
        if x.shape!=(3,):raise ValueError('one physical system per Oracle normalization record required')
        key=tuple(np.asarray(theta,float).tolist())
        if key in self.seen_tuples:raise ValueError('physical system repeated under another identity')
        self.oracle.add(x);self.seen_systems.add(system_key);self.seen_tuples.add(key)
        self.oracle_digest.update(system_key.encode()+x.astype('<f8').tobytes())

    def export(self):
        value=dict(schema=SCHEMA,source_split='train',horizons=list(HORIZONS),
            target_definition='physical_state_delta_8d',units=['m']*4+['m/s']*4,
            target_floor=FLOORS.tolist(),target={str(h):self.targets[h].export(FLOORS) for h in HORIZONS},
            oracle_coordinates=['log_m','gamma','log_k'],oracle=self.oracle.export(np.full(3,1e-6)),oracle_floor=1e-6,
            target_source_sha256=self.target_digest.hexdigest(),oracle_source_sha256=self.oracle_digest.hexdigest(),
            definition='population standard deviation; targets per declared training case/horizon, Oracle once per physical training system')
        value['statistics_sha256']=_digest(value)
        return value


class AHeadStatistics:
    def __init__(self,record):
        self.record=copy.deepcopy(record);self._check()
        self.sha256=self.record['statistics_sha256']

    def _check(self):
        r=self.record
        fields={'schema','source_split','horizons','target_definition','target','units','target_floor','oracle_coordinates','oracle','oracle_floor',
                'target_source_sha256','oracle_source_sha256','definition','statistics_sha256'}
        if set(r)!=fields or r['schema']!=SCHEMA or r['source_split']!='train' or r['horizons']!=list(HORIZONS):
            raise ValueError('unregistered A head normalization record')
        expected=_digest({k:v for k,v in r.items() if k!='statistics_sha256'})
        if r['statistics_sha256']!=expected:raise ValueError('A head normalization content changed')
        if r['units']!=['m']*4+['m/s']*4 or r['target_floor']!=FLOORS.tolist() or r['target_definition']!='physical_state_delta_8d':
            raise ValueError('A target units/floors differ')
        if r['oracle_coordinates']!=['log_m','gamma','log_k'] or r['oracle_floor']!=1e-6:
            raise ValueError('Oracle normalization coordinates differ')
        if set(r['target'])!={str(h) for h in HORIZONS}:raise ValueError('missing A head target horizon')
        for value,dim,floor in [(r['target'][str(h)],8,FLOORS) for h in HORIZONS]+[(r['oracle'],3,np.full(3,1e-6))]:
            if set(value)!={'count','mean','scale'} or type(value['count']) is not int or value['count']<2:raise ValueError('invalid training moment count')
            mean=np.asarray(value['mean']);scale=np.asarray(value['scale'])
            if mean.shape!=(dim,) or scale.shape!=(dim,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<floor):
                raise ValueError('invalid training-only normalization moments')

    def _target(self,values,horizon,inverse):
        self._check()
        if type(horizon) is not int or horizon not in HORIZONS:raise ValueError('unregistered A head target horizon')
        x=np.asarray(values,np.float64)
        if x.shape[-1:]!=(8,) or not np.isfinite(x).all():raise ValueError('invalid A head target/prediction')
        r=self.record['target'][str(horizon)];mean=np.asarray(r['mean']);scale=np.asarray(r['scale'])
        with np.errstate(over='ignore',invalid='ignore'):
            result=(x*scale+mean if inverse else (x-mean)/scale).astype(np.float32)
        if not np.isfinite(result).all():raise ValueError('A head normalized/physical output overflow')
        return result

    def standardize(self,delta,horizon):return self._target(delta,horizon,False)
    def physical(self,prediction,horizon):return self._target(prediction,horizon,True)
    def oracle_slot(self,theta):
        self._check();x=parameter_coordinates(theta);r=self.record['oracle']
        with np.errstate(over='ignore',invalid='ignore'):value=(x-np.asarray(r['mean']))/np.asarray(r['scale'])
        slot=np.zeros((*value.shape[:-1],128),np.float32)
        with np.errstate(over='ignore',invalid='ignore'):slot[...,:3]=value
        if not np.isfinite(slot).all():raise ValueError('Oracle normalized coordinate overflow')
        return slot
