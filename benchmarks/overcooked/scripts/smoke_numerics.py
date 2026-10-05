"""Check the actual random-pairing, probe and history-intervention functions without JAX."""
import ast
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'scripts')]
from probe_arrays import evaluate
from history_intervention import blank,batch_from_query,pair_for_episode,donor_index
source=ROOT/'variants/random/native_a/train.py'
tree=ast.parse(source.read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='relation_derangement')
ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
rows=[{'partner_identity':str(i%20)} for i in range(1024)]
a=ns['relation_derangement'](np.random.default_rng(17),rows)
b=ns['relation_derangement'](np.random.default_rng(17),rows)
np.testing.assert_array_equal(a,b)
assert sorted(a.tolist())==list(range(1024))
assert all(rows[i]['partner_identity']!=rows[int(j)]['partner_identity'] for i,j in enumerate(a))
try:ns['relation_derangement'](np.random.default_rng(17),[{'partner_identity':'one'}]*4)
except ValueError:pass
else:raise AssertionError('Single-identity relation should fail closed')
labels=np.repeat(np.arange(22),8);contexts=np.tile(np.arange(8),22)
fit=contexts<4;test=~fit;roles=np.where(labels<20,'train','heldout')
rng=np.random.default_rng(19);identity=rng.normal(size=(22,32));features=identity[labels]+rng.normal(size=(176,32))*0.001
targets=identity@rng.normal(size=(32,20))
result=evaluate(dict(features=features,labels=labels,contexts=contexts,fit=fit,test=test,roles=roles,targets=targets))
assert result['identity']['probe_test']['accuracy']==1
assert result['primary']['macro_mse']<result['primary']['mean_baseline_macro_mse']
bank={(i,e):blank(100) for i in (20,21) for e in range(8)}
for (i,e),v in bank.items():v['obs'][:]=i;v['attention_mask'][:]=1
query=[{'obs':np.ones((5,5,40)),'prev_actions':0,'prev_rewards':0.0}]
batch,length=batch_from_query(query,bank,20,donor_index(20),pair_for_episode(9))
assert length==1 and pair_for_episode(9)==(2,3)
assert np.array_equal(batch['query']['obs'][0],batch['query']['obs'][2])
assert batch['support']['attention_mask'][1].sum()==0
assert (batch['support']['obs'][0]==20).all() and (batch['support']['obs'][2]==21).all()
print('PASS: deterministic different-identity derangement, fit-only ridge probes, fixed-query history intervention.')
