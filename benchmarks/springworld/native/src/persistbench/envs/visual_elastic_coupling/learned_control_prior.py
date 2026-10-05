"""Frozen visual history readout feeding the common causal control estimator.

All learned families use the same nonlinear probe and virtual-measurement
calibration recipe. This is an empirical Gaussian log-parameter measurement,
not an exact posterior or a claim of optimal belief-space control.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .formation_readout import FrozenMLPReadout
from .formation import SELECTION_STRATA,LABELS
from .persistent_reference import posterior_samples


def calibrate(features,probes,output):
    from .dataset_snapshot import stable_digest
    from .training_protocol import source_fingerprint
    features=Path(features);probes=Path(probes);output=Path(output)
    if output.exists():raise ValueError('control readout calibration attempt exists')
    inputs=[features/'EXTRACTION.json',features/'FEATURES.private.npz',probes/'PROBE_REPORT.json',probes/'mlp.pt']
    before={str(p):stable_digest(p)['sha256'] for p in inputs};source_before=source_fingerprint()
    extraction=json.loads((features/'EXTRACTION.json').read_text());report=json.loads((probes/'PROBE_REPORT.json').read_text())
    digest=hashlib.sha256((features/'FEATURES.private.npz').read_bytes()).hexdigest()
    if extraction.get('test_read') or report.get('test_read') or extraction.get('random_initialization_control') or extraction['method']!='neural':
        raise ValueError('control readout needs the trained developmental representation')
    if extraction['frames']!=96 or extraction['feature_sha256']!=digest or report['features_sha256']!=digest or tuple(report['labels'])!=LABELS:
        raise ValueError('probe/feature calibration bindings differ')
    with np.load(features/'FEATURES.private.npz',allow_pickle=False) as data:
        if np.any(~np.isin(data['split'],['train','validation'])):raise ValueError('test rows forbidden in control readout calibration')
        select=(data['split']=='validation')&np.isin(data['stratum'],SELECTION_STRATA)
        x=data['features'][select];truth=data['labels'][select,:3];systems=data['system_key'][select]
    if len(np.unique(systems))<2:raise ValueError('insufficient independent calibration systems')
    readout=FrozenMLPReadout(probes/'mlp.pt');prediction=readout.predict(x)[:,:3]
    weights=np.zeros(len(systems))
    for key in np.unique(systems):
        mask=systems==key;weights[mask]=1/(len(np.unique(systems))*mask.sum())
    residual=truth-prediction;bias=np.sum(weights[:,None]*residual,axis=0);centered=residual-bias
    covariance=(centered*weights[:,None]).T@centered+np.eye(3)*.05**2
    result=dict(schema='vec.learned-control-prior.v1.1',status='DEVELOPMENT_CALIBRATED',family=extraction['family'],frames=96,
        source_fingerprint=source_before,extraction_sha256=before[str(features/'EXTRACTION.json')],probe_report_sha256=before[str(probes/'PROBE_REPORT.json')],
        checkpoint_sha256=extraction['checkpoint_sha256'],mlp_sha256=readout.sha256,feature_sha256=digest,
        bank_manifest_sha256=extraction['bank_manifest_sha256'],representation='128D mean/max persistent history memory',
        readout='same frozen two-layer128 GELU MLP formation probe; first3 log-physical outputs',
        bias=bias.tolist(),covariance=covariance.tolist(),log_std_floor=.05,systems=len(np.unique(systems)),histories=len(x),
        recipe='system-balanced validation residual mean and covariance plus fixed(.05)^2 I; virtual Gaussian log-parameter measurement',
        physical_prior_bounds=[[.5,.25,4.],[2.,1.5,25.]],
        limitations='validation also selected the pretraining checkpoint and probe; calibration residuals are not independent coverage evidence; formal held-out control assesses utility',
        formal_results=False,test_read=False)
    if before!={str(p):stable_digest(p)['sha256'] for p in inputs} or source_fingerprint()!=source_before:raise ValueError('control calibration inputs changed during fitting')
    output.mkdir(parents=True);(output/'CONTROL_PRIOR.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('family','systems','histories','bias','covariance')}),flush=True)


class LearnedParameterPrior:
    def __init__(self,model,readout,calibration):
        from .pixel_agent import PixelMemoryAgent
        if calibration.get('schema')!='vec.learned-control-prior.v1.1' or calibration['family']!=model.cfg.family or readout.sha256!=calibration['mlp_sha256']:
            raise ValueError('frozen control readout binding mismatch')
        self.memory=PixelMemoryAgent(model);self.readout=readout;self.calibration=calibration
        covariance=np.asarray(calibration['covariance'],float);self.bias=np.asarray(calibration['bias'],float)
        if covariance.shape!=(3,3) or self.bias.shape!=(3,) or not np.isfinite(covariance).all() or not np.isfinite(self.bias).all():raise ValueError('invalid virtual measurement')
        if not np.allclose(covariance,covariance.T) or np.linalg.eigvalsh(covariance).min()<=0:raise ValueError('nonpositive virtual measurement covariance')
        self.precision=np.linalg.inv(covariance);self._clear()

    def _clear(self):self.H=np.zeros((3,3));self.b=np.zeros(3);self.point=np.array([1.25,.875,14.5])
    def initialize(self,context):self.memory.initialize(context);self._clear();self.seed=int(context.seed)
    def reset(self,context):self.memory.reset(context)

    def ingest(self,experience):
        if self.memory.count or len(experience.observations)!=96:raise ValueError('control readout supports exactly one96-frame history')
        self.memory.ingest(experience);memory,_=self.memory.memory()
        log_measurement=self.readout.predict(memory.detach().cpu().numpy())[0,:3]+self.bias
        self.H=self.precision.copy();self.b=self.H@log_measurement
        self.point=posterior_samples(self.H,self.b,samples=256,seed=self.seed).mean(0)

    def prior(self):return self.H.copy(),self.b.copy(),self.point.copy()
    def mutable_state_bytes(self):return self.memory.mutable_state_bytes()+self.H.nbytes+self.b.nbytes+self.point.nbytes
    def frozen_fingerprint(self):
        digest=hashlib.sha256()
        for prefix,module in (('history',self.memory.model),('probe',self.readout.model)):
            for name,tensor in sorted(module.state_dict().items()):
                digest.update((prefix+'/'+name).encode()+tensor.detach().cpu().contiguous().numpy().tobytes())
        for value in (self.precision,self.bias,self.readout.xmean,self.readout.xscale,self.readout.xactive,self.readout.ymean,self.readout.yscale):
            digest.update(np.asarray(value).tobytes())
        return digest.hexdigest()


def load_prior(checkpoint,probes,calibration,model_source):
    import torch
    from .pixel_models import PixelDynamicsModel,PixelModelConfig
    from .pixel_training import runtime_policy
    checkpoint=Path(checkpoint);probes=Path(probes);calibration=Path(calibration);model_source=Path(model_source)
    config=json.loads(calibration.read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=config['checkpoint_sha256']:raise ValueError('control checkpoint differs from calibrated readout')
    for name in ('pixel_models.py','pixel_training.py'):
        original=model_source/'src/persistbench/envs/visual_elastic_coupling'/name
        if hashlib.sha256(original.read_bytes()).digest()!=hashlib.sha256((Path(__file__).parent/name).read_bytes()).digest():
            raise ValueError('control encoding/preprocessing differs from original model source')
    artifact=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if artifact['config']['bank_manifest_sha256']!=config['bank_manifest_sha256']:raise ValueError('control calibration bank differs from trained checkpoint')
    runtime_policy(artifact['config']['seed']);torch.set_num_threads(1)
    model=PixelDynamicsModel(PixelModelConfig(**artifact['config']['model']));model.load_state_dict(artifact['model'],strict=True)
    model.eval().requires_grad_(False)
    return LearnedParameterPrior(model,FrozenMLPReadout(probes/'mlp.pt'),config)


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--probes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();calibrate(a.features,a.probes,a.output)


if __name__=='__main__':main()
