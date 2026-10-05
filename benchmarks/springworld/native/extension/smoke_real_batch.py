"""One technical forward/backward on actual development images per A model form."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from persistent_jepa.losses import SIGReg
from strict_model import VisualBatch,StrictVisualJEPA,strict_objective


def load_batch(bank):
    history=[];past=[];targets=[];future=[];masks=[]
    # Two pairs, donor first half and recipient second half; same theta,
    # distinct episodes with legal, complete observation/target supports.
    for index in (9,21,10,22):
        p=np.load(bank/'episodes'/f'{index:05d}'/'visible_f2_r64.npz',allow_pickle=False)
        image=p['images'].astype(np.float32)/255
        difference=np.zeros_like(image);difference[1:]=image[1:]-image[:-1]
        tokens=np.stack((image,difference),axis=1)
        history.append(tokens[:24]);past.append(p['actions'][:23]);targets.append(tokens[np.array([24,27,39])])
        fa=np.zeros((3,16,2),np.float32);am=np.zeros((3,16),np.float32)
        for hi,h in enumerate((1,4,16)):fa[hi,:h]=p['actions'][23:23+h];am[hi,:h]=1
        future.append(fa);masks.append(am)
    return VisualBatch(*[torch.from_numpy(np.stack(x)) for x in (history,past,targets,future,masks)])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.output.exists():raise ValueError('use a fresh receipt')
    data=load_batch(args.bank);rows=[]
    for variant in ('B0','B0_split','B2','Bx','B3'):
        torch.manual_seed(610)
        model=StrictVisualJEPA(variant).train();model.begin_train_step()
        loss,metrics=strict_objective(model,data,SIGReg(),lambda_p=1.,lambda_x=.1)
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads)
        model.finish_train_step()
        rows.append(dict(variant=variant,loss=float(loss),parameters=sum(p.numel() for p in model.parameters()),
            context_dimensions=128,persistent_dimensions=0 if variant=='B0' else 64,
            forward_backward='PASS',history_only_statistics_updates=int(model.observation.norm.num_batches_tracked),
            losses={k:float(v) for k,v in metrics.items() if k.startswith('loss_')}))
        print(json.dumps(rows[-1]),flush=True)
    args.output.write_text(json.dumps(dict(status='PASS',technical_smoke_only=True,formal_training=False,test_read=False,
        input_source=str(args.bank),pairing='same theta, distinct episodes; donor first half -> recipient second half',
        raw_history_frames=24,observed_history_actions=23,target_horizons=[1,4,16],
        loss_weights=dict(sigreg=.02,alignment=1.,cross=.1),sigreg_directions=1024,models=rows),indent=2)+'\n')


if __name__=='__main__':main()
