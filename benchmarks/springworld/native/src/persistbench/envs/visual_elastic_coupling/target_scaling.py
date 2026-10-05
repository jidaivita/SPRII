"""Evaluator-owned physical target statistics, fitted on training labels only."""
import hashlib,json
import numpy as np

HORIZONS=(1,4,16,32)


class TargetStatistics:
    def __init__(self):
        self.count={h:0 for h in HORIZONS}
        self.mean={h:np.zeros(8) for h in HORIZONS}
        self.m2={h:np.zeros(8) for h in HORIZONS}
        self.digest=hashlib.sha256();self.seen=set()

    def add(self,*,split,episode_key,horizon,targets):
        # Refuse before inspecting the labels. Validation/test must not affect
        # even the shape, support, or numeric floors of training normalization.
        if split!='train':raise ValueError('target statistics accept train labels only')
        if horizon not in HORIZONS:raise ValueError('unregistered target horizon')
        key=(episode_key,horizon)
        if key in self.seen:raise ValueError('duplicate training target statistics')
        values=np.asarray(targets,float).reshape(-1,8)
        if not len(values) or not np.isfinite(values).all():raise ValueError('invalid training targets')
        n=len(values);old=self.count[horizon];delta=values.mean(0)-self.mean[horizon]
        self.mean[horizon]+=delta*n/(old+n)
        self.m2[horizon]+=np.sum((values-values.mean(0))**2,axis=0)+delta**2*old*n/(old+n)
        self.count[horizon]+=n;self.seen.add(key)
        self.digest.update(json.dumps(key,separators=(',',':')).encode()+values.astype('<f8').tobytes())

    def export(self):
        if any(n<2 for n in self.count.values()):raise ValueError('insufficient training labels per horizon')
        floors=np.array([1e-5]*4+[1e-4]*4)
        return dict(schema='vec.train-target-statistics.v1.1',source_split='train',
            source_sha256=self.digest.hexdigest(),unit_order=['m']*4+['m/s']*4,
            mean={str(h):self.mean[h].tolist() for h in HORIZONS},
            scale={str(h):np.maximum(np.sqrt(self.m2[h]/self.count[h]),floors).tolist() for h in HORIZONS},
            count={str(h):self.count[h] for h in HORIZONS},floors=floors.tolist(),
            definition='population standard deviation of physical state deltas, separately by horizon; train labels only')
