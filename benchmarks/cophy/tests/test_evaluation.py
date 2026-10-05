import json
import unittest
import numpy as np
import torch
from test_revision import ROOT
from cophy_evaluate import formation_report
from cophy_summarize import summarize, intervals
from cophy_training import mse_per_recipient


class EvaluationTests(unittest.TestCase):
    def test_uncovered_input_is_not_zero_error_or_model_based_exclusion(self):
        target=torch.zeros(2,3,1,3);pred=torch.ones_like(target)
        result=mse_per_recipient(pred,target,torch.tensor([[1.],[0.]]),3,True)
        self.assertEqual(result[0].item(),1.)
        self.assertTrue(torch.isnan(result[1]))
        pred[1,0,0,0]=float('nan')
        with self.assertRaises(ValueError):mse_per_recipient(pred,target,torch.tensor([[1.],[0.]]),3,True)

    def test_probe_reports_known_signal_and_uses_disjoint_experiments(self):
        rng=np.random.default_rng(14)
        def data(prefix,n):
            codes={};rows=[]
            for i in range(n):
                ident=f'{prefix}{i}';label=i%2
                u=rng.normal(size=(1,32));u[0,0]=20*(2*label-1)
                codes[ident]=(u,np.ones(1))
                rows.append({'id':ident,'slot':0,'known_type':'ball','physical':[label]})
            return codes,rows
        train,tr=data('train',100);evaluation,ev=data('val',40)
        result=formation_report(train,evaluation,tr,ev,['mass'])
        self.assertEqual(result['fields']['mass']['P']['balanced_accuracy'],1.)
        self.assertEqual(result['fields']['mass']['slot_type_prior']['balanced_accuracy'],.5)
        self.assertEqual(result['inner_fit_experiments'],80)
        json.dumps(result,allow_nan=False)
        with self.assertRaises(ValueError):formation_report(train,train,tr,tr,['mass'])

    def test_paired_reductions_average_focals_within_each_recipient(self):
        # r1 has two focal objects; it must not count twice as much as r2.
        reports=[]
        for method,errors in [('Native',[4.,4.,8.]),('A',[2.,2.,7.]),('Random',[3.,3.,8.])]:
            rows=[]
            for (ident,slot),error in zip([('r1',0),('r1',1),('r2',0)],errors):
                rows.append({'recipient':ident,'focal':slot,'errors':{arm:{'focal_mse':error+delta,'scene_mse':error+delta}
                    for arm,delta in [('Correct',0),('Null',1),('Wrong-any',2)]}})
            reports.append({'scene':'balls','split':'val','seed':0,'preflight_sha256':'same',
                'method':method,'official':{'model':float(np.mean(errors))},
                'official_rows':[{'id':'r1','model':errors[0],'CopyC':10.,'CopyC_gtmask':10.},
                                 {'id':'r2','model':errors[-1],'CopyC':10.,'CopyC_gtmask':10.}],
                'assays':{'primary':{'manifest_sha256':'same-primary','coverage':{},'rows':rows},
                          'wrong1':{'manifest_sha256':'same-factor','coverage':{},'rows':[]}}})
        result=summarize(reports)
        paired=result['assays']['primary']['results']['focal_mse']['paired']
        self.assertEqual(paired['ReuseGain']['estimate'],1.5)
        self.assertEqual(paired['DoD_Wrong_any']['estimate'],0.)
        ci=intervals([[2.,4.],[3.,6.]],['r1','r2'],repeats=100)
        np.testing.assert_allclose(np.array(ci[0]['ci95'])*2,ci[1]['ci95'])


if __name__=='__main__':unittest.main()
