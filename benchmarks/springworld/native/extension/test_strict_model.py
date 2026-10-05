import copy
import unittest
import torch
from persistent_jepa.losses import SIGReg
from strict_model import StrictVisualJEPA,VisualBatch,strict_objective


def batch():
    torch.manual_seed(802)
    history=torch.rand(4,24,2,64,64);history[:,0,1]=0
    actions=torch.rand(4,23,2)*.1
    target=torch.rand(4,3,2,64,64)
    future=torch.zeros(4,3,16,2);masks=torch.zeros(4,3,16)
    for hi,h in enumerate((1,4,16)):
        future[:,hi,:h]=.1;masks[:,hi,:h]=1
    return VisualBatch(history,actions,target,future,masks)


def prediction(model,data):
    torch.manual_seed(713)
    history,target=model.encode_batch(data)
    _,_,context=model.codes(history,data.history_actions)
    pred=model.predictor(context,data.future_actions[:,2],data.action_masks[:,2],torch.full((4,),2,dtype=torch.long))
    return history,target,pred


class StrictBoundary(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(912)
        self.model=StrictVisualJEPA('B3').train();self.data=batch()

    def test_future_targets_do_not_change_current_forward_or_statistics(self):
        other=copy.deepcopy(self.model);changed=copy.deepcopy(self.data);changed.target_images.fill_(17.)
        self.model.begin_train_step();other.begin_train_step()
        h1,t1,p1=prediction(self.model,self.data);h2,t2,p2=prediction(other,changed)
        torch.testing.assert_close(h1,h2,atol=0,rtol=0);torch.testing.assert_close(p1,p2,atol=0,rtol=0)
        self.assertGreater(float((t1-t2).abs().max()),.01)
        self.model.finish_train_step();other.finish_train_step()
        torch.testing.assert_close(self.model.observation.norm.running_mean,other.observation.norm.running_mean,atol=0,rtol=0)
        torch.testing.assert_close(self.model.observation.norm.running_var,other.observation.norm.running_var,atol=0,rtol=0)

    def test_other_case_history_does_not_change_this_case_forward(self):
        other=copy.deepcopy(self.model);changed=copy.deepcopy(self.data)
        changed.history_images[2:]*=3
        self.model.begin_train_step();other.begin_train_step()
        h1,_,p1=prediction(self.model,self.data);h2,_,p2=prediction(other,changed)
        torch.testing.assert_close(h1[:2],h2[:2],atol=0,rtol=0)
        torch.testing.assert_close(p1[:2],p2[:2],atol=0,rtol=0)

    def test_original_objective_backpropagates_through_targets(self):
        self.data.target_images.requires_grad_(True)
        self.model.begin_train_step()
        before=self.model.observation.norm.running_mean.clone()
        loss,metrics=strict_objective(self.model,self.data,SIGReg(num_directions=16),lambda_p=1.,lambda_x=.1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(self.data.target_images.grad.abs().sum()),0)
        self.assertGreater(float(self.model.observation.cnn[0].weight.grad.abs().sum()),0)
        torch.testing.assert_close(before,self.model.observation.norm.running_mean,atol=0,rtol=0)
        self.model.finish_train_step()
        self.assertEqual(int(self.model.observation.norm.num_batches_tracked),1)
        self.assertGreater(float((before-self.model.observation.norm.running_mean).abs().sum()),0)
        self.assertIn('loss_cross',metrics);self.assertIn('loss_persist',metrics)

    def test_evaluation_cannot_update_statistics(self):
        self.model.eval();before=self.model.observation.norm.running_mean.clone()
        prediction(self.model,self.data)
        torch.testing.assert_close(before,self.model.observation.norm.running_mean,atol=0,rtol=0)
        with self.assertRaises(RuntimeError):self.model.begin_train_step()
        self.model.train();self.model.begin_train_step();self.model.encode_batch(self.data)
        with self.assertRaises(ValueError):self.model.finish_train_step(split='validation')


if __name__=='__main__':unittest.main()
