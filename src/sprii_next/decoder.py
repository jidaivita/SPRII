"""Train-only physical bottleneck of the identical P64 used by Persistent."""
from dataclasses import dataclass
import numpy as np


def physical_coordinates(theta, dim=3):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != 3 or not np.isfinite(theta).all() or np.any(theta <= 0):
        raise ValueError('positive finite [mass, drag, stiffness] required')
    z = np.log(theta)
    if dim == 4:
        z = np.column_stack((z, z[:, 2] - z[:, 0]))
    elif dim != 3:
        raise ValueError('only Decode3 or secondary Decode4 supported')
    return z


@dataclass
class RidgeDecoder:
    mean_x: np.ndarray
    scale_x: np.ndarray
    mean_y: np.ndarray
    scale_y: np.ndarray
    coef: np.ndarray
    alpha: float
    fit_systems: tuple

    @classmethod
    def fit(cls, p, theta, systems, *, split, alpha=1., dim=3):
        if split != 'train':
            raise PermissionError('physics decoder fitting is train-only')
        p = np.asarray(p, np.float64)
        if p.ndim != 2 or p.shape[1] != 64 or not np.isfinite(p).all():
            raise ValueError('decoder input must be the frozen Persistent P64')
        systems = np.asarray(systems)
        if len(p) != len(theta) or len(p) != len(systems) or alpha <= 0:
            raise ValueError('invalid decoder population or ridge coefficient')
        # Oracle/probe target moments count each physical system once.
        ids, first, counts = np.unique(systems, return_index=True, return_counts=True)
        for sid, pos in zip(ids, first):
            if not np.all(np.asarray(theta)[systems == sid] == np.asarray(theta)[pos]):
                raise ValueError('parameters change within physical system')
        y = physical_coordinates(theta, dim)
        my, sy = y[first].mean(0), y[first].std(0).clip(1e-6)
        # Equal weight per system, independent of donor-window count.
        count_map = dict(zip(ids, counts))
        w = np.asarray([1 / count_map[s] for s in systems])
        w *= len(p) / w.sum()
        mx = np.average(p, axis=0, weights=w)
        sx = np.sqrt(np.average((p-mx)**2, axis=0, weights=w)).clip(1e-6)
        x = (p-mx)/sx
        target = (y-my)/sy
        # Intercept is required: source codes and labels can have weighted means.
        x = np.column_stack((x, np.ones(len(x))))
        reg = np.eye(65) * alpha
        reg[-1, -1] = 0.
        coef = np.linalg.solve(x.T @ (w[:, None]*x) + reg, x.T @ (w[:, None]*target))
        return cls(mx, sx, my, sy, coef, float(alpha), tuple(ids.tolist()))

    def predict(self, p):
        x = (np.asarray(p, np.float64)-self.mean_x)/self.scale_x
        return (np.column_stack((x, np.ones(len(x)))) @ self.coef).astype(np.float32)

    def oracle(self, theta):
        return ((physical_coordinates(theta, len(self.mean_y))-self.mean_y)/self.scale_y).astype(np.float32)

    def record(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(self).items()}

    @classmethod
    def from_record(cls, record):
        r = dict(record)
        for k in ('mean_x', 'scale_x', 'mean_y', 'scale_y', 'coef'):
            r[k] = np.asarray(r[k], np.float64)
        r['fit_systems'] = tuple(r['fit_systems'])
        return cls(**r)

    def score(self, p, theta, systems):
        if set(systems) & set(self.fit_systems):
            raise ValueError('probe scoring systems overlap train')
        y = physical_coordinates(theta, len(self.mean_y))
        pred = self.predict(p)*self.scale_y + self.mean_y
        ids,first,counts=np.unique(systems,return_index=True,return_counts=True)
        count_map=dict(zip(ids,counts));w=np.asarray([1/count_map[s] for s in systems])
        residual = np.square(pred-y)
        mean=np.average(y,axis=0,weights=w)
        mse=np.average(residual,axis=0,weights=w)
        denom=np.average(np.square(y-mean),axis=0,weights=w)
        ratio_target=y[:,2]-y[:,0];ratio_prediction=pred[:,2]-pred[:,0]
        ratio_mse=np.average((ratio_prediction-ratio_target)**2,weights=w)
        ratio_var=np.average((ratio_target-np.average(ratio_target,weights=w))**2,weights=w)
        return dict(r2=(1-mse/np.maximum(denom, 1e-20)).tolist(),
                    coordinate_names=['log_m','log_gamma','log_k']+(['log_k_over_m'] if len(self.mean_y)==4 else []),
                    derived_log_k_over_m_r2=float(1-ratio_mse/max(ratio_var,1e-20)),
                    mse=mse.tolist(), coordinates='log_physics', fit_split='train',aggregation='equal physical systems',
                    systems=len(set(systems)), windows=len(p))
