"""Inspect the resumable 500-step CPC pilot; source-only, fixed train clips."""
import importlib.util,json,sys,math
from pathlib import Path
import torch
source=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('_cpc_calibration_train',source)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
args=m.parser().parse_args(sys.argv[2:]);data,b,planner,model=m.setup(args)
folder=Path(args.out)/'runs'/args.method;ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
if ck['step']<500 or ck['binding_sha256']!=b['sha256']:raise ValueError('Pilot checkpoint missing or wrong source binding')
model.load_state_dict(ck['model'],strict=True);diag=m.source_diagnostic(model,data)
if not all(math.isfinite(v) for k,v in diag.items() if k!='selection'):raise FloatingPointError('Nonfinite pilot diagnostics')
if diag['target_std']<1e-6:raise FloatingPointError('Effectively constant CPC target')
sm=m.read(folder/'smoke.json');result=dict(status='CALIBRATION_COMPLETE',step=ck['step'],method=args.method,scene=args.scene,source_checkpoint_sha256=m.digest(folder/'latest.pt'),before_motion_response=sm['motion_response'],before_motion_spread=sm['spread'],after=diag,criterion='finite source computation; target not numerically constant; no downstream win requirement',test_read=False)
m.write(folder/'calibration_complete.json',result);m.emit('CALIBRATION_COMPLETE',**result)
