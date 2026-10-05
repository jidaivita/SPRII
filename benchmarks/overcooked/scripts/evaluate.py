"""Frozen online evaluation: 20 episodes per partner, episodes 6--20 primary."""
import argparse
import ast
import copy
import functools
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True,help='Native terminal checkpoint descriptor')
p.add_argument('--train-manifest',type=Path,required=True)
p.add_argument('--heldout-manifest',type=Path,required=True)
p.add_argument('--qualification',type=Path,required=True,help='Policy selection output with verified tensor hashes')
p.add_argument('--probe-plan',type=Path,help='Frozen common sample plan; required for the GPU equivalence gate')
p.add_argument('--out',type=Path,required=True)
p.add_argument('--seed',type=int,default=920140)
p.add_argument('--backend',choices=('cpu','gpu'),default='cpu')
a = p.parse_args()
os.environ['JAX_PLATFORMS']='cuda,cpu' if a.backend=='gpu' else 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
sys.path[:0]=[str(a.repo.resolve()),str(Path(__file__).resolve().parents[1]/'tools')]
import jax
import jax.numpy as jnp
import numpy as np
from benchmarks.manifest_schema import TaskEntry
from native_a.train import load_native_checkpoint,tensor_sha
from history_intervention import host_only_partner_restore
jax.config.update('jax_default_matmul_precision','highest')
model,params,cfg,checkpoint=load_native_checkpoint(a.checkpoint.resolve())
assert checkpoint['step']==20000 and cfg['seed'] in (4200,4201,4202)
initial=tensor_sha(params)
qualified=json.loads(a.qualification.read_text())
expected={r['task']['task_id']:r['parameter_sha256'] for r in qualified['selected']+qualified['development']}
if a.backend=='cpu':
    from native_a.evaluate import evaluate_native_task
else:
    if a.probe_plan is None:
        p.error('GPU evaluation requires --probe-plan for the original CPU/GPU action equivalence gate')
    from frozen_probe import panel_support_batch
    from native_a.model import apply_batch
    panel=json.loads(a.probe_plan.read_text())
    sample=next(s for s in panel['samples'] if s['partner_role']=='train' and s['probe_split']=='probe_fit')
    support=panel_support_batch([sample])
    def logits(batch):
        output=apply_batch(model,params,batch,train=False)
        n=batch['query']['attention_mask'].sum(axis=1).astype(jnp.int32)
        return output['logits'][jnp.arange(n.shape[0]),n-1]
    cpu_infer=jax.jit(logits,backend='cpu');gpu_infer=jax.jit(logits,backend='gpu')
    for length in (1,17,100,300):
        query={k:np.concatenate([v[:,0]]*3,axis=1) for k,v in support.items()}
        query['attention_mask'][:]=0;query['attention_mask'][:,:length]=1
        batch={'query':query,'support':support}
        c,g=map(np.asarray,(cpu_infer(batch),gpu_infer(batch)))
        np.testing.assert_allclose(c,g,rtol=2e-5,atol=2e-5)
        assert np.array_equal(c.argmax(-1),g.argmax(-1))
    # This is the same two-decorator dispatch adapter used in the final evaluation.
    source=a.repo/'native_a/evaluate.py';tree=ast.parse(source.read_text())
    nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef)
           and n.name in ('evaluate_native_task','baseline_batch_jax')]
    guard=nodes[0].body[1]
    assert isinstance(guard,ast.If) and ast.unparse(guard.test)=="jax.default_backend() != 'cpu'"
    nodes[0].body.remove(guard)
    changed=0
    for node in ast.walk(nodes[0]):
        if isinstance(node,ast.FunctionDef) and node.name=='infer':
            assert len(node.decorator_list)==1 and ast.unparse(node.decorator_list[0])=='jax.jit'
            node.decorator_list=[ast.parse("functools.partial(jax.jit, backend='gpu')",mode='eval').body]
            changed+=1
    assert changed==2
    namespace={'jax':jax,'jnp':jnp,'np':np,'functools':functools}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),namespace)
    evaluate_native_task=namespace['evaluate_native_task']
results=[]
for role,path,n in (('familiar',a.train_manifest,20),('heldout_development',a.heldout_manifest,2)):
    tasks=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(tasks)==n
    for raw in tasks:
        task=TaskEntry.from_json(dict(raw,split='train'));saved=[]
        with jax.default_device(jax.devices('cpu')[0]),host_only_partner_restore():
            evaluate_native_task(model,params,task,mode=cfg['mode'],episodes=20,seed=a.seed,
                                 history_condition='matched',episode_callback=saved.append,
                                 expected_teammate_params_sha256=expected[raw['task_id']])
        assert len(saved)==20
        values=np.array([r['return'] for r in saved])
        results.append({'task_id':raw['task_id'],'role':role,'mean_return_primary':float(values[5:].mean()),
                        'mean_return_all20':float(values.mean()),
                        'positive_return_rate_primary':float((values[5:]>0).mean())})
        jax.clear_caches()
assert tensor_sha(params)==initial
out={'partner_results':results,'frozen_encoder_updates':0,'checkpoint_params_sha256':initial,
     'aggregates':{role:{k:float(np.mean([r[k] for r in results if r['role']==role]))
                        for k in ('mean_return_primary','mean_return_all20','positive_return_rate_primary')}
                   for role in ('familiar','heldout_development')}}
a.out.write_text(json.dumps(out,indent=2)+'\n')
