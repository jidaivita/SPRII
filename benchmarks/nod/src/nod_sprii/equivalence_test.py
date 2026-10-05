"""Deterministic lambda_align=0 equivalence test for the Burgers fork."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F

from train_nod_clean import NGS_INR, indices_to_coords_and_targets, sample_query_indices, set_seed
from train_sprii_clean import relation_loss


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())


def state_diff(a: dict, b: dict) -> float:
    vals=[]
    for k in a:
        x,y=a[k],b[k]
        if torch.is_tensor(x): vals.append(max_diff(x,y))
        elif isinstance(x,dict): vals.append(state_diff(x,y))
    return max(vals, default=0.0)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=10)
    ap.add_argument('--tolerance',type=float,default=1e-6)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    set_seed(20260922)
    kw=dict(context_dim=1,cond_t=5,cond_x=8,pred_x=8,deeponet_latent_dim=8,
            predictor_branch_hidden=12,predictor_branch_layers=2,predictor_trunk_hidden=12,
            predictor_trunk_layers=2,conditioner_base_channels=2,conditioner_head_hidden=8,
            pe_num_freqs_x=2,pe_num_freqs_t=1)
    nod=NGS_INR(**kw); sprii=NGS_INR(**kw); sprii.load_state_dict(nod.state_dict())
    on=torch.optim.AdamW(nod.parameters(),lr=1e-3,weight_decay=1e-4)
    os=torch.optim.AdamW(sprii.parameters(),lr=1e-3,weight_decay=1e-4)
    cond=torch.randn(2,1,5,8); align=torch.randn(2,1,5,8); u0=torch.randn(2,1,8)
    t_idx=torch.arange(5).view(1,5).repeat(2,1); target=torch.randn(2,5,1,8)
    metadata={'sample_ids':[101,202],'cond_case_idx':[3,7],'pred_case_idx':[11,13],
              'align_case_idx':[19,23],'normalization':{'mode':'none'}}
    rows=[]
    for step in range(args.steps):
        on.zero_grad(set_to_none=True); os.zero_grad(set_to_none=True)
        torch.manual_seed(5000+step)
        za=nod.encode(cond)
        idx=sample_query_indices(2,40,10,target.device)
        c1,y1=indices_to_coords_and_targets(target,t_idx,idx,4.0)
        p1=nod.predict_queries(u0,c1,za); l1=F.mse_loss(p1,y1)
        l1.backward(); on.step()
        torch.manual_seed(5000+step)
        za=sprii.encode(cond); zc=sprii.encode(align)
        idx=sample_query_indices(2,40,10,target.device)
        c2,y2=indices_to_coords_and_targets(target,t_idx,idx,4.0)
        p2=sprii.predict_queries(u0,c2,za); base=F.mse_loss(p2,y2)
        rel=relation_loss(za,zc,False); l2=base+0.0*rel
        l2.backward(); os.step()
        row={'step':step,'loss_diff':abs(float(l1)-float(l2)),'prediction_max_abs_diff':max_diff(p1,p2),
             'gradient_max_abs_diff':state_diff({k:v.grad for k,v in nod.named_parameters()},
                                                 {k:v.grad for k,v in sprii.named_parameters()}),
             'optimizer_state_max_abs_diff':state_diff(on.state_dict(),os.state_dict()),
             'parameter_max_abs_diff':state_diff(nod.state_dict(),sprii.state_dict()),
             'relation_loss_value':float(rel.detach())}
        rows.append(row)
    max_seen=max((max(x.values()) for x in rows if isinstance(x,dict) for k in x if k.endswith('diff')),default=0.0)
    passed=all(all(float(v)<=args.tolerance for k,v in r.items() if k.endswith('diff')) for r in rows)
    out={'passed':passed,'tolerance':args.tolerance,'steps':args.steps,'metadata':metadata,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    raise SystemExit(0 if passed else 1)

if __name__=='__main__': main()
