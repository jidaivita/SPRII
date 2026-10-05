"""Inference-only access to the frozen formation probes; no label arguments."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


class FrozenMLPReadout:
    def __init__(self,path):
        import torch
        from torch import nn
        self.torch=torch;self.sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        artifact=torch.load(path,map_location='cpu',weights_only=True)
        self.xmean=np.asarray(artifact['xmean']);self.xscale=np.asarray(artifact['xscale']);self.xactive=np.asarray(artifact['xactive'],bool)
        self.ymean=np.asarray(artifact['ymean']);self.yscale=np.asarray(artifact['yscale'])
        if not np.isfinite(np.r_[self.xmean,self.xscale,self.ymean,self.yscale]).all() or np.any(self.xscale<=0) or np.any(self.yscale<=0):
            raise ValueError('invalid frozen probe normalization')
        if self.xactive.shape!=self.xmean.shape or self.xscale.shape!=self.xmean.shape or self.xactive.sum()!=artifact['input_dim']:
            raise ValueError('invalid frozen probe feature support')
        self.model=nn.Sequential(nn.Linear(artifact['input_dim'],128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,artifact['output_dim']))
        self.model.load_state_dict(artifact['model'],strict=True);self.model.eval().requires_grad_(False)

    def predict(self,features):
        values=np.asarray(features,float)
        if values.ndim!=2 or values.shape[1]!=len(self.xmean) or not np.isfinite(values).all():raise ValueError('invalid frozen probe features')
        x=((values-self.xmean)/self.xscale)[:,self.xactive]
        with self.torch.inference_mode():prediction=self.model(self.torch.as_tensor(x,dtype=self.torch.float32)).numpy()
        result=prediction*self.yscale+self.ymean
        if not np.isfinite(result).all():raise ValueError('nonfinite frozen probe prediction')
        return result


def ridge_predict(path,features):
    with np.load(path,allow_pickle=False) as artifact:
        x=np.asarray(features,float)
        if x.ndim!=2 or x.shape[1]!=len(artifact['xmean']) or not np.isfinite(x).all():raise ValueError('invalid frozen ridge features')
        standardized=((x-artifact['xmean'])/artifact['xscale'])[:,artifact['xactive']]
        return (standardized@artifact['weights'])*artifact['yscale']+artifact['ymean']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--features',type=Path,required=True);parser.add_argument('--probes',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise ValueError('probe prediction attempt exists')
    # Loading only the features key is deliberate: a label-bearing bundle can
    # remain evaluator-owned, but the frozen predictor never reads its labels.
    with np.load(a.features,allow_pickle=False) as bundle:features=bundle['features']
    mlp=FrozenMLPReadout(a.probes/'mlp.pt');predictions=dict(mlp=mlp.predict(features),ridge=ridge_predict(a.probes/'ridge.npz',features))
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'PREDICTIONS.npz',**predictions)
    (a.output/'PREDICTION_RECEIPT.json').write_text(json.dumps(dict(status='EXECUTED',labels_read=False,optimization_performed=False,
        feature_file_sha256=hashlib.sha256(a.features.read_bytes()).hexdigest(),mlp_sha256=mlp.sha256,
        ridge_sha256=hashlib.sha256((a.probes/'ridge.npz').read_bytes()).hexdigest(),rows=len(features)),indent=2)+'\n')


if __name__=='__main__':main()
