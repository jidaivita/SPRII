"""Reconstruct the executed Uniform50 pair indices without any optimization/rendering."""
import hashlib,json,pathlib,sys,types,time
R=pathlib.Path(os.environ.get('SPRII_ROOT', '.'));S=R/'benchmarks/pokeworld/revision'
sys.path.insert(0,str(S/'src'))
import numpy as np
import torch
from persistent_jepa import poke_torch as mod
torch.set_num_threads(1)
D=R/'data/pokeworld_factorized'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=R/'poke_uniform50_empirical_match_20260925.json';assert not out.exists()
    started=time.time();d=mod.PokeSplit(D,'train');orig=d._from_indices;cls=mod.PokeBatch
    refs={}
    for seed,step in [(0,1),(1,100),(2,20000)]:
        b,mask=d.fidelity_paired_batch(48,seed*10000000+step,20260901,step,.5)
        refs[(seed,step)]=(b,mask)
    def metadata(self,systems,rollouts,anchors):
        return types.SimpleNamespace(**{k:torch.from_numpy(np.asarray(v).copy()) for k,v in dict(
             mass=self.mass[systems],gamma=self.gamma[systems],stiffness=self.stiffness[systems],
             system_index=systems,rollout_id=rollouts,anchor=anchors).items()})
    d._from_indices=types.MethodType(metadata,d);mod.PokeBatch=types.SimpleNamespace
    for (seed,step),(reference,reference_mask) in refs.items():
        b,mask=d.fidelity_paired_batch(48,seed*10000000+step,20260901,step,.5)
        assert np.array_equal(mask,reference_mask)
        for k,v in b.__dict__.items():assert torch.equal(v,getattr(reference,k)),k
    result=[]
    for seed in range(3):
        cfg=json.loads((R/f'poke_structured_v1/uniform50_s{seed}/config.json').read_text())
        assert cfg['fidelity_q']==.5 and cfg['steps']==20000 and cfg['fidelity_assignment_seed']==20260901
        counts=np.zeros(3,np.int64);wrong_counts=np.zeros(3,np.int64);different=0
        for step in range(1,20001):
            b,mask=d.fidelity_paired_batch(48,seed*10000000+step,20260901,step,.5)
            matches=np.stack([(getattr(b,k)[:48]==getattr(b,k)[48:]).numpy() for k in ['mass','gamma','stiffness']],axis=1)
            assert int(mask.sum())==24 and matches[mask].all()
            assert (b.system_index[:48][~mask]!=b.system_index[48:][~mask]).all()
            counts+=matches.sum(0);wrong_counts+=matches[~mask].sum(0)
        assert np.array_equal(counts,wrong_counts+480000)
        result.append(dict(seed=seed,steps=20000,pairs=960000,incorrect_pair_slots=480000,
             matched_counts=counts.tolist(),match_rates=(counts/960000).tolist(),wrong_slot_matched_counts=wrong_counts.tolist(),wrong_slot_match_rates=(wrong_counts/480000).tolist(),
             config_sha256=sha(R/f'poke_structured_v1/uniform50_s{seed}/config.json')))
    mod.PokeBatch=cls;d._from_indices=orig
    total=np.sum([v['matched_counts'] for v in result],axis=0)
    res=dict(status='PASS',factors=['mass','drag','stiffness'],seeds=result,total_pairs=2880000,
        pooled_match_rates=(total/2880000).tolist(),source_sampler_sha256=sha(S/'src/persistent_jepa/poke_torch.py'),
        bank_manifest_sha256=sha(D/'manifest.json'),audit='Exact original fidelity_paired_batch over all executed 3x20000 sampler seeds; only data materialization replaced by metadata. Three real-batch index/factor/mask equality checks PASS.',
        source_optimization_updates=0,new_environment_interactions=0,test_read=False,seconds=time.time()-started)
    out.write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res),flush=True)
if __name__=='__main__':main()
