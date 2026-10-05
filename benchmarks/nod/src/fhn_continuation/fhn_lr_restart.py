"""Explicit RMSprop learning-rate restart; preserve moments and StepLR state."""
import argparse
import sys
import torch
from fhn_minimal import train as core

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--restart-lr',required=True,type=float)
a,remaining=p.parse_known_args()
assert 0<a.restart_lr<=.001
original_load=torch.optim.RMSprop.load_state_dict
original_save=core.atomic_save
def load(self,state):
    result=original_load(self,state)
    for group in self.param_groups:group['lr']=a.restart_lr
    return result
def save(payload,path):
    payload['restart_lr']=a.restart_lr
    payload['resume_data_policy']='seeded stream restart; RMSprop moments and StepLR state preserved; group LR reset'
    original_save(payload,path)
torch.optim.RMSprop.load_state_dict=load
core.atomic_save=save
sys.argv=[sys.argv[0]]+remaining
core.main()
