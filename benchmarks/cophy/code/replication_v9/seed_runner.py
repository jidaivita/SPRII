"""Explicit, hash-bound seed replication; never edits published parent code."""
import argparse,hashlib,json,os,sys,time,types
from pathlib import Path

def patch(source,kind,seed):
    edits=[]
    def replace(old,new,count=1):
        nonlocal source
        if source.count(old)!=count:raise ValueError('Parent changed: '+old)
        source=source.replace(old,new);edits.append(dict(old=old,new=new,count=count))
    if kind in ('mono','cross'):
        replace('choices=(0,),default=0','choices=(0,1,2),default=0')
    elif kind in ('cpc','rssm'):
        replace('args.seed!=0','args.seed not in (0,1,2)')
    elif kind=='base':
        replace('1 <= args.epochs <= 50','1 <= args.epochs <= 100')
    elif kind=='readout':
        replace("seedof('v6-", "replica_module_seed('v6-",5)
        replace('seed=0, selection_ids_sha256=',f'seed={seed}, replica_runner_sha256=REPLICA_SHA, selection_ids_sha256=')
        replace('torch.manual_seed(991+epoch)',f'torch.manual_seed(991+epoch+{seed}*100000)')
        replace('np.random.default_rng(771+epoch)',f'np.random.default_rng(771+epoch+{seed}*100000)')
    elif kind=='controlled':
        replace("module.__file__=str(core);module.CONTROL=dict(recipe)","module.__file__=str(core);module.CONTROL=dict(recipe)")
        replace("    name='_controlled_'", "    replace('seed=0, selection_ids_sha256=', 'seed='+str(REPLICA_SEED)+', selection_ids_sha256=')\n    replace('torch.manual_seed(991+epoch)', 'torch.manual_seed(991+epoch+'+str(REPLICA_SEED*100000)+')')\n    replace('np.random.default_rng(771+epoch)', 'np.random.default_rng(771+epoch+'+str(REPLICA_SEED*100000)+')')\n    name='_controlled_'")
        replace("    def module_seed(label):\n", "    def module_seed(label):\n        if REPLICA_SEED:\n            return int.from_bytes(hashlib.sha256(('replica:'+str(REPLICA_SEED)+':'+recipe['initialization']+':'+label).encode()).digest()[:8],'little')\n")
        replace('source+Path(__file__).read_text()', 'source+Path(__file__).read_text()+REPLICA_SHA')
    else:raise ValueError(kind)
    return source,edits

def main():
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--sha',required=True);p.add_argument('--kind',required=True);p.add_argument('--seed',type=int,choices=(0,1,2),required=True);p.add_argument('--receipt',required=True);p.add_argument('--check-only',action='store_true');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    original=Path(a.original).resolve();raw=original.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.sha:raise ValueError('Parent SHA mismatch')
    source,edits=patch(raw.decode(),a.kind,a.seed);args=a.arguments
    if args and args[0]=='--':args=args[1:]
    if a.kind not in ('readout','controlled'):
        if '--parent-checkpoint' in args:raise ValueError('Replicas must start from random initialization')
        if '--seed' in args:
            if int(args[args.index('--seed')+1])!=a.seed:raise ValueError('Seed mismatch')
        else:args+=['--seed',str(a.seed)]
    runner_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    r=dict(version='cophy-v9-seed-replication-1',original=str(original),original_sha256=a.sha,effective_source_sha256=hashlib.sha256(source.encode()).hexdigest(),runner_sha256=runner_sha,seed=a.seed,kind=a.kind,edits=edits,source_pretrained_checkpoint=None if a.kind not in ('readout','controlled') else 'frozen specified source',test_read=False)
    rp=Path(a.receipt);rp.parent.mkdir(parents=True,exist_ok=True)
    if rp.exists() and json.loads(rp.read_text())!=r:raise ValueError('Replica binding changed')
    tmp=rp.with_name(rp.name+'.tmp');tmp.write_text(json.dumps(r,indent=2));os.replace(tmp,rp)
    compile(source,str(original),'exec')
    if a.check_only:print(json.dumps(dict(status='PASS',kind=a.kind,seed=a.seed,edits=len(edits))));return
    sys.path.insert(0,str(original.parent));sys.argv=[str(original),*args]
    m=types.ModuleType('__main__');m.__file__=str(original);m.REPLICA_SHA=runner_sha;m.REPLICA_SEED=a.seed
    def module_seed(label):
        text=label if a.seed==0 else 'replica:'+str(a.seed)+':'+label
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')
    m.replica_module_seed=module_seed;sys.modules['__main__']=m
    exec(compile(source,str(original),'exec'),m.__dict__)
if __name__=='__main__':main()
