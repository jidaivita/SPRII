import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from cophy_manifests import qualify_objects, build_donor_manifests, parameter_documents


class ManifestTests(unittest.TestCase):
    def test_contrasts_are_paired_and_wrong1_does_not_shrink_primary_queue(self):
        raw=[]; cache={}
        # Two replicas per property class permit Correct; only two diagonal
        # classes exist, so Wrong-any exists but no single-factor Wrong-1 does.
        for i,physical in enumerate(([0,0],[0,0],[1,1],[1,1])):
            ident=str(i)
            raw.append(dict(id=ident,slot=0,split='val',known_type='ball',physical=physical,in_C=True))
            cache[ident]={'cache_version':'ab_c_float32_v2','presence_ab':np.ones(1),'presence_c':np.ones(1)}
        rows=qualify_objects(raw,cache,scene='balls',split='val',field_indices=[0,1],gravity_branch='not_applicable')
        primary,wrong1=build_donor_manifests(rows,scene='balls',split='val',physical_fields=['mass','friction'])
        self.assertEqual(len(primary['rows']),4)
        self.assertEqual(len(wrong1['rows']),0)
        lookup={r['id']:r['physical'] for r in rows}
        for row in primary['rows']:
            self.assertNotEqual(row['recipient'],row['Correct']['id'])
            self.assertEqual(lookup[row['recipient']],lookup[row['Correct']['id']])
            self.assertNotEqual(lookup[row['recipient']],lookup[row['Wrong-any']['id']])
        self.assertEqual(primary,build_donor_manifests(rows[::-1],scene='balls',split='val',physical_fields=['mass','friction'])[0])
        with self.assertRaises(ValueError):
            build_donor_manifests(rows,scene='balls',split='test',physical_fields=['mass','friction'])

    def test_gravity_strata_and_visual_coverage_are_explicit(self):
        raw=[];cache={}
        for i in range(8):
            ident=str(i)
            raw.append(dict(id=ident,slot=0,split='val',known_type='block',physical=[(i//2)%2],
                            raw_gravity=[-.5 if i<4 else .5,0.],in_C=True))
            cache[ident]={'cache_version':'ab_c_float32_v2','presence_ab':np.ones(1),'presence_c':np.ones(1)}
        cache['0']['presence_c']=np.zeros(1)
        rows=qualify_objects(raw,cache,scene='blocktower',split='val',field_indices=[0],gravity_branch='varying_verified')
        primary,_=build_donor_manifests(rows,scene='blocktower',split='val',physical_fields=['mass'])
        self.assertEqual(primary['coverage']['candidate_opportunities'],8)
        self.assertEqual(primary['coverage']['visually_eligible'],7)
        for row in primary['rows']:
            for arm in ['Correct','Wrong-any']:
                self.assertEqual(int(row['recipient'])//4,int(row[arm]['id'])//4)
        with self.assertRaises(ValueError):
            qualify_objects(raw,cache,scene='blocktower',split='val',field_indices=[0],gravity_branch='unreliable')

    def test_parameter_statistics_use_training_only_and_include_global_gravity(self):
        def row(split,ident,mass,g):
            return dict(id=ident,split=split,slot=0,raw_physical=[mass],raw_gravity=[g,0.])
        docs=parameter_documents({'train':[row('train','0',1,-.5),row('train','1',3,.5)],
                                  'val':[row('val','2',100,99)]},scene='blocktower',slots=4,object_fields=['mass'],include_gravity=True)
        self.assertEqual(docs['train']['fields'],docs['val']['fields'])
        self.assertEqual([f['name'] for f in docs['train']['fields']],['mass','gravity_x','gravity_y'])
        self.assertEqual(docs['val']['fields'][0]['train_mean'],2.)
        self.assertEqual(docs['val']['fields'][1]['train_mean'],0.)


if __name__=='__main__': unittest.main()
