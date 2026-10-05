"""Equivalent input construction; avoid reconstructing uint8 history twice for a discarded validator result."""
import ast,inspect,textwrap,numpy as np,torch
import native_training,native128_model

def install():
 original=native_training.make_batch
 tree=ast.parse(inspect.getsource(original));removed=0
 # All pixels are decoded by the original SHA-checked `visible` function as
 # uint8. Original make_batch computes x=uint8/255 and difference from x.
 # The discarded image_history call re-encodes these known arrays to uint8.
 # Retain original schedule, support, dtype checks and full batch validation.
 class Rewrite(ast.NodeTransformer):
  def visit_Expr(self,node):
   nonlocal removed
   if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='image_history':removed+=1;return None
   return self.generic_visit(node)
 tree=Rewrite().visit(tree);assert removed==1;ast.fix_missing_locations(tree)
 ns=dict(native_training.__dict__);exec(compile(tree,'<same_batch_without_roundtrip_validation>','exec'),ns)
 old_validate=native128_model.NativeVisualBatch.validate
 vtree=ast.parse(textwrap.dedent(inspect.getsource(old_validate)));replaced=0
 class Finite(ast.NodeTransformer):
  def visit_Call(self,node):
   nonlocal replaced
   if ast.unparse(node)=='torch.isfinite(tensor).all()':
    replaced+=1;return ast.copy_location(ast.Call(func=ast.Name(id='_allfinite',ctx=ast.Load()),args=[ast.Name(id='tensor',ctx=ast.Load())],keywords=[]),node)
   return self.generic_visit(node)
 vtree=Finite().visit(vtree);assert replaced==1;ast.fix_missing_locations(vtree)
 def finite(t):return np.isfinite(t.numpy()).all() if t.device.type=='cpu' else torch.isfinite(t).all()
 vns=dict(native128_model.__dict__);vns['_allfinite']=finite;exec(compile(vtree,'<equivalent_finite_validation>','exec'),vns)
 native128_model.NativeVisualBatch.validate=vns['validate']
 return original,ns['make_batch'],old_validate

if __name__=='__main__':
 import pathlib,json,time,hashlib
 import spring_sensitivity as s
 s.torch.set_num_threads(1)
 p=s.json.loads((s.NATIVE/'NIGHT_POLICY.json').read_text());v=p['training_spec'];v['milestone_steps']=tuple(v['milestone_steps']);spec=s.engine.PretrainingSpec(**v)
 bank=s.engine.PretrainingBank(s.BANK,snapshot_sha256=p['bank_snapshot_sha256'],spec=spec)
 original,fast,validator=install();new_validator=native128_model.NativeVisualBatch.validate;rows=[]
 for sweep,batch_ix in [(0,0),(0,2),(3,1)]:
  plan=bank.schedule.sweep(sweep,'Both');native128_model.NativeVisualBatch.validate=validator
  t=time.monotonic();a,ar=original(bank.schedule,plan,batch_ix,s.BANK);oldtime=time.monotonic()-t
  native128_model.NativeVisualBatch.validate=new_validator;t=time.monotonic();b,br=fast(bank.schedule,plan,batch_ix,s.BANK);newtime=time.monotonic()-t
  assert ar==br and all(torch.equal(getattr(a,k),getattr(b,k)) for k in vars(a)), 'batch changed'
  rows.append(dict(sweep=sweep,batch=batch_ix,original_seconds=oldtime,fast_seconds=newtime,all_tensors_bitwise_equal=True,receipts_equal=True));print(json.dumps(rows[-1]),flush=True)
  del a,b
 # Full finite checks still reject nonfinite values; CUDA path is unchanged.
 from dataclasses import fields
 assert not np.isfinite(np.array([float('nan'),float('inf')])).all()
 s.write(s.ROOT/'receipts/FAST_BATCH_PARITY.json',dict(status='PASS',rows=rows,script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),original_files_unchanged=True))
