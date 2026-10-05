import ast
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import cophy_protocol as policy

class FakeDerenderer(nn.Module):
    def __init__(self,num_objects=2):
        super().__init__();self.num_objects=num_objects
        self.bn=nn.BatchNorm1d(1)
    def forward(self,x):
        b=len(x);k=self.num_objects
        v=x.mean((1,2,3))
        presence=torch.ones(b,k,device=x.device)
        presence[:,-1]=-1
        pose=v[:,None,None].expand(b,k,3).clone()
        return presence,pose,torch.zeros(b,k,4)

def load_defs(rel,names,extra=None):
    tree=ast.parse((ROOT/'code'/rel).read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    namespace={'torch':torch,'nn':nn,'F':F,'np':np,'time':time,'tqdm':lambda x:x,'DeRendering':FakeDerenderer}
    namespace.update(extra or {})
    exec(compile(ast.Module(body=selected,type_ignores=[]),rel,'exec'),namespace)
    return namespace

model=load_defs('cf_learning/model.py',{'aggreg_E','CoPhyNet','CopyC','extract_pose_ab_c'})
main=load_defs('cf_learning/main.py',{'get_losses','get_mse','validate','get_dataloaders'})
util=load_defs('dataloaders/utils.py',{'get_stab'})

class RevisionTests(unittest.TestCase):
    def test_all_categories_and_rejections(self):
        for support in ([1,2,5],[.1,.5,1],[1,10]):
            self.assertEqual([policy.category_index(x,support) for x in support],list(range(len(support))))
        self.assertEqual(policy.category_index(.0997877,[.1,.5,1]),0)
        for value,levels in [(float('nan'),[1]),(3,[1,2,5]),(.105,[.1,.11])]:
            with self.assertRaises(ValueError):policy.category_index(value,levels)

    def test_stationary_and_moving_labels(self):
        pose=np.zeros((6,2,3));pose[:,1,0]=np.arange(6)
        labels=util['get_stab'](pose,np.ones(2),t_delta=1)
        np.testing.assert_array_equal(labels[:,0],np.ones(6))
        np.testing.assert_array_equal(labels[:,1],np.zeros(6))
        np.testing.assert_array_equal(util['get_stab'](pose,np.array([1,0]))[:,1],np.zeros(6))

    def test_loss_cannot_hide_errors_with_predictions(self):
        pred=torch.ones(1,3,2,3,requires_grad=True)
        gt=torch.zeros_like(pred);active=torch.tensor([[1.,0.]])
        stationary=torch.zeros(1,3,2)
        a=main['get_losses'](pred,torch.full((1,3,2),100.),torch.zeros(1,2),gt,stationary,active)[1][1]
        b=main['get_losses'](pred,torch.full((1,3,2),-100.),torch.ones(1,2),gt,stationary,active)[1][1]
        self.assertEqual(a.item(),1);self.assertEqual(b.item(),1)
        a.backward();self.assertGreater(pred.grad[:,:,0].abs().sum(),0)
        self.assertEqual(pred.grad[:,:,1].abs().sum(),0)
        with self.assertRaises(ValueError):main['get_losses'](pred,stationary,active,gt,stationary,torch.zeros_like(active))

    def test_stationary_gate_and_update_schedule(self):
        net=model['CoPhyNet'](2).eval()
        for p in net.parameters():nn.init.zeros_(p)
        nn.init.ones_(net.fc_delta.bias)
        u=torch.zeros(1,2,32);c=torch.zeros(1,2,3);presence=torch.ones(1,2)
        count=[]
        def stable(*args):count.append(1);return torch.full((1,2,1),100.)
        net.pred_stab=stable
        out,_=net.pred_D(u,c,presence,T=3)
        self.assertEqual(len(count),3);self.assertEqual(out.abs().sum(),0)
        net.pred_stab=lambda *args:torch.full((1,2,1),-100.)
        out,_=net.pred_D(u,c,presence,T=3)
        torch.testing.assert_close(out[0,:,0,0],torch.tensor([1.,2.,3.]))

    def test_derenderer_frozen_eval_and_input_cache_equivalence(self):
        net=model['CoPhyNet'](2);net.train()
        self.assertFalse(net.derendering.training)
        ab=torch.rand(2,4,3,4,4);c=torch.rand(2,1,3,4,4)
        pa,pc,xa,xc=model['extract_pose_ab_c'](net.derendering,ab,c)
        net.eval()
        live=net(ab,c)[0]
        cached=net(None,None,pa,xa,pc,xc)[0]
        torch.testing.assert_close(live,cached,rtol=0,atol=0)
        with self.assertRaises(ValueError):model['extract_pose_ab_c'](net.derendering,ab,ab)
        copier=model['CopyC'](2)
        torch.testing.assert_close(copier(ab,c)[0],copier(None,None,pa,xa,pc,xc)[0])

    def test_metric_dimensions_and_empty_presence(self):
        pred=torch.zeros(1,3,2,3);pred[:,:,:,2]=3;gt=torch.zeros_like(pred);mask=torch.ones(1,2)
        self.assertEqual(main['get_mse'](pred,gt,mask,D=2).item(),0)
        self.assertEqual(main['get_mse'](pred,gt,mask,D=3).item(),3)
        with self.assertRaises(ValueError):main['get_mse'](pred,gt,torch.zeros_like(mask))

    def test_loader_never_instantiates_test_during_training(self):
        calls=[]
        class Dataset:
            def __init__(self,**kwargs):
                calls.append(kwargs['split']);self.num_objects=4
                if kwargs['split']=='test':raise AssertionError('test touched')
        class Loader:
            def __init__(self,dataset,**kwargs):self.dataset=dataset
        ns=load_defs('cf_learning/main.py',{'get_dataloaders'},dict(Balls_CF=Dataset,Collision_CF=Dataset,Blocktower_CF=Dataset,DataLoader=Loader))
        for name in ['balls','blocktower','collision']:
            calls.clear();tr,val,te,d=ns['get_dataloaders'](name,'unused',{},evaluate_on_test_only=False)
            self.assertEqual(calls,['train','val']);self.assertIsNone(te)
            self.assertEqual(d,2 if name=='balls' else 3)

    def test_sample_weighted_validation_not_batch_weighted(self):
        from types import SimpleNamespace
        class Net:
            def eval(self): pass
            def __call__(self, *args):
                xa, mask = args[3], args[4]
                return xa, mask, None
        class Loader(list): pass
        def sample(error,n):
            return {'pred_pose_3D_ab':torch.full((n,1,1,3),error**.5),
                    'pred_pose_3D_cd':torch.zeros(n,1,1,3),
                    'pred_presence_ab':torch.ones(n,1),'pred_presence_cd':torch.ones(n,1),
                    'pose_3D_cd':torch.zeros(n,2,1,3)}
        loader=Loader([sample(1,2),sample(9,1)]);loader.dataset=SimpleNamespace(is_rgb=False)
        with tempfile.TemporaryDirectory() as temp:
            score=main['validate'](Net(),'cpu',loader,temp,str(Path(temp)/'val.txt'),D=2)
        self.assertAlmostEqual(score,11/3,places=5)

    def test_cache_contains_only_ab_and_c(self):
        import types
        from unittest.mock import patch
        fake=types.ModuleType('cf_learning.model')
        fake.extract_pose_ab_c=model['extract_pose_ab_c']
        cache_code=load_defs('derendering/extract_object_visual_properties.py',{'extract_object_visual_properties'})
        ab=torch.rand(1,4,3,4,4);cd=torch.rand(1,4,3,4,4)
        inp={'id':['example'], 'rgb_ab':ab, 'rgb_cd':cd}
        with patch.dict(sys.modules,{'cf_learning.model':fake}):
            first=cache_code['extract_object_visual_properties'](FakeDerenderer(2),'cpu',[inp])
            cd[:,1:]+=1000
            second=cache_code['extract_object_visual_properties'](FakeDerenderer(2),'cpu',[inp])
        self.assertEqual(set(first['example']),{'cache_version','presence_ab','presence_c','pose_ab','pose_c'})
        for name in ['presence_ab','presence_c','pose_ab','pose_c']:
            np.testing.assert_array_equal(first['example'][name],second['example'][name])
            self.assertEqual(first['example'][name].dtype,np.float32)

    def test_vicreg_singletons_and_gradients(self):
        x=torch.randn(8,16,requires_grad=True);y=torch.randn(8,16,requires_grad=True)
        loss,parts=policy.vicreg_focal(x,y);self.assertTrue(torch.isfinite(loss))
        loss.backward();self.assertGreater(x.grad.abs().sum(),0);self.assertGreater(y.grad.abs().sum(),0)
        z,parts=policy.vicreg_focal(x[:1],y[:1]);self.assertEqual(z.item(),0);self.assertTrue(parts['skipped'])
        with self.assertRaises(ValueError):policy.vicreg_focal(x[:,0],y[:,0])

    def test_or_extension(self):
        self.assertTrue(policy.should_extend(1,1,1,.95))
        self.assertTrue(policy.should_extend(1,.95,1,1))
        self.assertFalse(policy.should_extend(1,1,1,1))

    def test_dod_not_mistaken_for_reuse_gain(self):
        r=policy.paired_summary([1,1],[2,2],[2,2],[10,10],[3,3],[4,4],['a','b'],repeats=30)
        self.assertEqual(r['dod_wrong_any']['estimate'],7)
        self.assertEqual(r['reuse_gain']['estimate'],-1)
        self.assertEqual(r['a_vs_random_correct']['estimate'],2)

    def test_probe_nonconstant_labels(self):
        x=np.array([[-3],[-2],[-1],[1],[2],[3]])
        y=np.array([0,0,0,1,1,1])
        np.testing.assert_array_equal(policy.ridge_probe(x,y,np.array([[-2],[2]]),2),[0,1])

    def test_manifest_tampering_and_test_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'data').write_text('frozen')
            names=['audit','relation_index','sampler','random_rule','validation_correct','validation_wrong_any','validation_wrong_1','test_generator','protocol']
            doc={'status':'PASS','artifacts':{n:{'path':'data','sha256':policy.digest(p/'data')} for n in names}}
            receipt=p/'preflight.json';receipt.write_text(json.dumps(doc))
            policy.verify_preflight(receipt)
            with self.assertRaises(ValueError):policy.verify_preflight(receipt,require_release=True)
            (p/'data').write_text('changed')
            with self.assertRaises(ValueError):policy.verify_preflight(receipt)

if __name__=='__main__':unittest.main(verbosity=2)
