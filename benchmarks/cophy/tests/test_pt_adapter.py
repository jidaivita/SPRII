import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from test_revision import ROOT, model as original, load_defs, FakeDerenderer
from cophy_adapter import (ABObservation, VisualInput, Targets, DonorPairs, PTCoPhy,
                           split_code, replace_p, objective, probe_views, donor_assay, PersistentNullBank)
from cophy_relations import RelationIndex, PairProvider, ParameterFeatures
from cophy_protocol import digest, verify_adapter_binding, vicreg_focal
from cophy_training import train_adapter_epoch, validate_adapter

torch.set_num_threads(1)


def visual(b=4, k=2):
    return VisualInput(ABObservation(torch.randn(b, 5, k, 3), torch.ones(b, k)),
                       torch.randn(b, 1, k, 3), torch.ones(b, k))


def target(v):
    return Targets(v.c.repeat(1, 4, 1, 1)+.1, torch.zeros(len(v.c), 4, v.c.shape[2]), v.presence_c)


def records(n=4):
    return {'version': 'cophy-relation-index-v3', 'scene': 'collision', 'split': 'train',
            'records': [{'id': str(i), 'slot': 0, 'stratum': ['sphere', 0],
                         'physical': [i % 2, 1], 'recipient': True, 'donor': True} for i in range(n)]}


class TinyDataset:
    def __init__(self, n=4):
        self.list_ex = [str(i) for i in range(n)]
        self.num_objects = 2
        self.is_rgb = False
        rng = np.random.default_rng(123)
        self.dict_id2object_properties = {
            str(i): {'cache_version': 'ab_c_float32_v2', 'pose_ab': rng.normal(size=(5, 2, 3)).astype('float32'),
                     'presence_ab': np.ones(2, 'float32'), 'pose_c': rng.normal(size=(1, 2, 3)).astype('float32'),
                     'presence_c': np.ones(2, 'float32')} for i in range(n)}

    def __len__(self):
        return len(self.list_ex)

    def __getitem__(self, i):
        ident = self.list_ex[i]
        p = self.dict_id2object_properties[ident]
        return {'id': ident, 'pred_pose_3D_ab': p['pose_ab'], 'pred_presence_ab': p['presence_ab'],
                'pred_pose_3D_cd': p['pose_c'], 'pred_presence_cd': p['presence_c'],
                'pose_3D_cd': np.repeat(p['pose_c'], 5, axis=0)+.1,
                'presence_cd': np.ones(2, 'float32'), 'stab_cd': np.zeros((5, 2), 'float32')}


def features(split='train'):
    return {'version': 'cophy-parameters-v3', 'scene': 'collision', 'split': split,
            'all_varying_parameters_included': True,
            'fields': [{'name': 'mass', 'train_mean': 2., 'train_scale': 1.}],
            'examples': {str(i): [[float(i+1)], [2.]] for i in range(4)}}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_native_exactly_preserves_predictions_and_parameter_count(self):
        backbone = original['CoPhyNet'](2).eval()
        v = visual()
        before = backbone(None, None, v.ab.presence, v.ab.pose, v.presence_c, v.c)
        net = PTCoPhy(backbone, 'Native').eval()
        after = net(None, None, v.ab.presence, v.ab.pose, v.presence_c, v.c)
        for x, y in zip(before, after):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        for method in ['Native', 'A', 'Random']:
            model = PTCoPhy(copy.deepcopy(backbone), method)
            self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in backbone.parameters()))

    def test_cross_gradient_cannot_enter_donor_transient_or_erase_recipient_transient(self):
        recipient = torch.randn(3, 2, 32, requires_grad=True)
        donor = torch.randn(3, 2, 32, requires_grad=True)
        focal = torch.tensor([0, 1, 0])
        mixed = replace_p(recipient, focal, donor[torch.arange(3), focal, :16])
        torch.testing.assert_close(mixed[..., 16:], recipient[..., 16:], rtol=0, atol=0)
        for i, k in enumerate(focal):
            torch.testing.assert_close(mixed[i, 1-k], recipient[i, 1-k], rtol=0, atol=0)
        mixed.sum().backward()
        self.assertEqual(float(donor.grad[..., 16:].abs().sum()), 0.)
        self.assertGreater(float(donor.grad[..., :16].abs().sum()), 0.)
        torch.testing.assert_close(recipient.grad[..., 16:], torch.ones_like(recipient.grad[..., 16:]))

    def test_donor_transient_perturbation_leaves_predictions_unchanged(self):
        net = PTCoPhy(original['CoPhyNet'](2), 'A').eval()
        v = visual()
        recipient, donor = net.encode_ab(v.ab), torch.randn(4, 2, 32)
        focal = torch.tensor([0, 1, 0, 1]); rows = torch.arange(4)
        a = net.predict_code(replace_p(recipient, focal, donor[rows, focal, :16]), v)[0]
        donor[..., 16:] += 10000
        b = net.predict_code(replace_p(recipient, focal, donor[rows, focal, :16]), v)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_future_and_gt_metadata_not_read_by_encoder(self):
        net = PTCoPhy(original['CoPhyNet'](2), 'A')
        batch = next(iter(DataLoader(TinyDataset(), batch_size=4)))
        first = net.input_from_batch(batch, 'cpu')
        batch['pose_3D_cd'][:, 1:] += 90000
        batch['confounders'] = torch.tensor(float('nan'))
        second = net.input_from_batch(batch, 'cpu')
        torch.testing.assert_close(net.encode_ab(first.ab), net.encode_ab(second.ab), rtol=0, atol=0)
        batch['pred_pose_3D_cd'] = torch.zeros(4, 5, 2, 3)
        with self.assertRaises(ValueError): net.input_from_batch(batch, 'cpu')
        with self.assertRaises(ValueError): net.code_for_task(first, torch.zeros(4, 2, 3))

    def test_vicreg_only_p_and_old_32_dim_rejected(self):
        u = torch.randn(8, 32, requires_grad=True); d = torch.randn(8, 32, requires_grad=True)
        loss, _ = vicreg_focal(u[:, :16], d[:, :16]); loss.backward()
        self.assertEqual(float(u.grad[:, 16:].abs().sum()), 0.)
        self.assertGreater(float(u.grad[:, :16].abs().sum()), 0.)
        with self.assertRaises(ValueError): vicreg_focal(u, d)

    def test_param_known_keeps_recipient_transient_and_uses_all_features(self):
        net = PTCoPhy(original['CoPhyNet'](2), 'Param-known', parameter_dim=3)
        v = visual()
        original_u = net.encode_ab(v.ab)
        u = net.code_for_task(v, torch.randn(4, 2, 3))
        torch.testing.assert_close(u[..., 16:], original_u[..., 16:], rtol=0, atol=0)
        with self.assertRaises(ValueError): net.code_for_task(v, torch.zeros(4, 2, 2))
        with self.assertRaises(ValueError): probe_views(net, v)
        self.assertEqual(ParameterFeatures(features(), 'collision', 'train').batch(['0'], 2, 'cpu')[0, 0, 0], -1)

    def test_same_pair_eligibility_donor_multiset_and_no_self_episode(self):
        index = RelationIndex(records(), map(str, range(4)), 'collision')
        for epoch in range(10):
            plan, _ = index.plan(['0', '1', '2', '3'], 8, epoch, 0)
            self.assertEqual(len(plan), 4)
            self.assertEqual(sorted(r['correct']['id'] for r in plan), sorted(r['random']['id'] for r in plan))
            for r in plan:
                self.assertNotEqual(str(r['row']), r['correct']['id'])
                self.assertNotEqual(str(r['row']), r['random']['id'])
            np.random.seed(epoch*123)
            self.assertEqual(plan, index.plan(['0', '1', '2', '3'], 8, epoch, 0)[0])
        invalid = records(); invalid['split'] = 'test'
        with self.assertRaises(ValueError): RelationIndex(invalid, ['0'], 'collision')

    def test_provider_reads_only_donor_ab_cache_and_empty_pair_is_legal(self):
        class NoTargets(TinyDataset):
            def __getitem__(self, i): raise AssertionError('Donor target read')
        data = NoTargets()
        provider = PairProvider(RelationIndex(records(), data.list_ex, 'collision'), data, seed=0)
        v = visual()
        for method in ['A', 'Random']:
            pairs, _ = provider.make(data.list_ex, method, 1, 0, v)
            loss, _, stats = objective(PTCoPhy(original['CoPhyNet'](2), method), v, target(v), pairs)
            loss.backward(); self.assertEqual(stats['paired'], 4)
        # Rare strata remain usable even in a one-recipient minibatch.
        pairs, _ = provider.make(['0'], 'A', 1, 0, v.select(torch.tensor([0])))
        self.assertEqual(len(pairs.rows), 1)
        no_matches = records()
        for i, row in enumerate(no_matches['records']): row['physical'] = [i]
        empty_provider = PairProvider(RelationIndex(no_matches, data.list_ex, 'collision'), data, 0)
        pairs, _ = empty_provider.make(['0'], 'A', 1, 0, v.select(torch.tensor([0])))
        self.assertEqual(len(pairs.rows), 0)
        loss, _, stats = objective(PTCoPhy(original['CoPhyNet'](2), 'A'), v.select(torch.tensor([0])),
                                   target(v).select(torch.tensor([0])), pairs)
        self.assertEqual(stats['paired'], 0); loss.backward()

    def test_all_assay_arms_keep_recipient_t_and_probes_share_checkpoint(self):
        net = PTCoPhy(original['CoPhyNet'](2), 'A').eval(); v = visual()
        recorded = []; actual = net.predict_code
        def record(u, observed):
            recorded.append(u.clone()); return actual(u, observed)
        net.predict_code = record
        donors = {'Correct': (visual().ab, torch.zeros(4, dtype=torch.long)),
                  'Wrong-any': (visual().ab, torch.zeros(4, dtype=torch.long))}
        out = donor_assay(net, v, target(v), torch.zeros(4, dtype=torch.long), donors, torch.zeros(4, 16), 2)
        self.assertEqual(set(out), {'Own', 'Null', 'Null-zero', 'Correct', 'Wrong-any'})
        for u in recorded:
            torch.testing.assert_close(u[..., 16:], recorded[0][..., 16:], rtol=0, atol=0)
            torch.testing.assert_close(u[:, 1], recorded[0][:, 1], rtol=0, atol=0)
        views = probe_views(net, v)
        self.assertEqual(views['P'].shape[-1], 16); self.assertEqual(views['U'].shape[-1], 32)

    def test_all_four_methods_execute_training_and_validation(self):
        data = TinyDataset(); index = RelationIndex(records(), data.list_ex, 'collision')
        with tempfile.TemporaryDirectory() as temp:
            for method in ['Native', 'A', 'Random', 'Param-known']:
                model = PTCoPhy(original['CoPhyNet'](2), method, 1 if method == 'Param-known' else None)
                initial = {n: p.detach().clone() for n, p in model.named_parameters()}
                loader = DataLoader(data, batch_size=4)
                store = ParameterFeatures(features(), 'collision', 'train') if method == 'Param-known' else None
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                info = train_adapter_epoch(model, 'cpu', loader, optimizer, str(Path(temp)/'train'), epoch=1,
                                           pair_provider=PairProvider(index, data, 0) if method in {'A', 'Random'} else None,
                                           parameter_store=store)
                score = validate_adapter(model, 'cpu', loader, str(Path(temp)/'val'), parameter_store=store)
                self.assertTrue(np.isfinite(score)); self.assertEqual(info['updates'], 1)
                self.assertTrue(any(not torch.equal(initial[n], p) for n, p in model.named_parameters() if p.requires_grad))

    def test_three_scene_slot_and_horizon_interfaces(self):
        for scene, slots, frames, dims in [('collision', 4, 15, 3), ('balls', 9, 30, 2), ('blocktower', 4, 30, 3)]:
            with self.subTest(scene=scene):
                net = PTCoPhy(original['CoPhyNet'](slots), 'A')
                obs = VisualInput(ABObservation(torch.randn(2, frames, slots, 3), torch.ones(2, slots)),
                                  torch.randn(2, 1, slots, 3), torch.ones(2, slots))
                tar = Targets(torch.randn(2, frames-1, slots, 3), torch.zeros(2, frames-1, slots), torch.ones(2, slots))
                pairs = DonorPairs(torch.tensor([0, 1]), torch.tensor([0, slots-1]),
                                   ABObservation(obs.ab.pose.flip(0), obs.ab.presence), torch.tensor([0, slots-1]))
                loss, prediction, _ = objective(net, obs, tar, pairs)
                loss.backward()
                self.assertEqual(prediction[0].shape, tar.pose.shape)
                self.assertTrue(torch.isfinite(loss))

    def test_null_mean_ignores_absent_slots_and_rejects_other_checkpoint_or_split(self):
        bank = PersistentNullBank('checkpoint-1')
        p = torch.zeros(2, 2, 16); p[0, 0] = 2; p[1, 0] = 4; p[:, 1] = 10000
        keys = [[(0, 'ball'), (1, 'ball')]]*2
        bank.update(p, torch.tensor([[1, 0], [1, 0]]), keys, split='train')
        value = bank.lookup([(0, 'ball')], checkpoint_sha256='checkpoint-1', device='cpu')
        torch.testing.assert_close(value, torch.full((1, 16), 3.))
        with self.assertRaises(ValueError): bank.update(p, torch.ones(2, 2), keys, split='val')
        with self.assertRaises(ValueError): bank.lookup([(0, 'ball')], checkpoint_sha256='checkpoint-2', device='cpu')
        with self.assertRaises(ValueError): bank.lookup([(1, 'ball')], checkpoint_sha256='checkpoint-1', device='cpu')
        bank.update(p, torch.ones(2, 2), keys, split='train')
        second = bank.lookup([(1, 'ball')], checkpoint_sha256='checkpoint-1', device='cpu')
        torch.testing.assert_close(second, torch.full((1, 16), 10000.))
        with self.assertRaises(ValueError):
            bank.update(p, torch.ones(2, 2), [[(0, 'ball'), (0, 'ball')]]*2, split='train')

    def test_real_main_resume_exactly_matches_continuous_training(self):
        main = load_defs('cf_learning/main.py', {'main'}, {
            'CoPhyNet': original['CoPhyNet'], 'CopyC': original['CopyC'],
            'random': __import__('random'), 'os': __import__('os'), 'optim': torch.optim})['main']
        def loaders(*args, sampler_seed=0, **kwargs):
            data = TinyDataset()
            return (DataLoader(data, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(sampler_seed)),
                    DataLoader(data, batch_size=4, generator=torch.Generator().manual_seed(sampler_seed+1)), None, 3)
        main.__globals__['get_dataloaders'] = loaders
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); derender = root/'frontend.pt'; torch.save(FakeDerenderer(2).state_dict(), derender)
            artifacts = {}
            documents = {'audit': {'parameter_fields': ['mass']}, 'relation_index': records(),
                         'sampler': {}, 'random_rule': {}, 'validation_correct': {},
                         'validation_wrong_any': {}, 'validation_wrong_1': {}, 'test_generator': {}, 'protocol': {}}
            for name, value in documents.items():
                path = root/(name+'.json'); path.write_text(json.dumps(value))
                artifacts[name] = {'path': str(path), 'sha256': digest(path)}
            artifacts['derenderer'] = {'path': str(derender), 'sha256': digest(derender)}
            for split in ('train', 'val'):
                cache = root/f'collision_normal_{split}_extracted_prop.pickle'
                cache.write_bytes(b'synthetic loader fixture')
                artifacts[f'cache_{split}'] = {'path': str(cache), 'sha256': digest(cache)}
            preflight = root/'preflight.json'
            preflight.write_text(json.dumps({'status': 'PASS', 'adapter_version': 'cophy-pt16-v3', 'scene': 'collision',
                'allowed_methods': ['Native','A','Random','Param-known'],
                'input_profile': {'scene': 'collision', 'num_objects': 2, 'type': 'normal', 'dataset_dir': str(root)},
                'code_sha256': {str(p.relative_to(ROOT/'code')): digest(p) for p in sorted((ROOT/'code').rglob('*.py'))},
                'artifacts': artifacts}))
            args = SimpleNamespace(preflight_receipt=str(preflight), sealed_release=None, evaluate=False,
                derendering_ckpt=str(derender), method='A', train_from_rgb=False, model='cophynet', seed=3,
                batch_size=4, workers=0, dataset_name='collision', dataset_dir=str(root), num_objects=2, type='normal',
                preextracted_obj_vis_prop_dir=str(root), lambda_x=.1, lambda_p=1., resume=None,
                log_dir=str(root/'split'), epochs=1)
            main(args)
            args.resume = str(root/'split/latest_resume.pt'); args.epochs = 2; main(args)
            args.resume = None; args.log_dir = str(root/'continuous'); main(args)
            a = torch.load(root/'split/latest_resume.pt', weights_only=False)
            b = torch.load(root/'continuous/latest_resume.pt', weights_only=False)
            self.assertEqual(a['epoch'], 2)
            for name in a['model']: torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
            # Stale full-U v2 receipts are not admitted to this execution path.
            stale = json.loads(preflight.read_text()); stale['adapter_version'] = 'corrected-v2'
            preflight.write_text(json.dumps(stale))
            with self.assertRaises(ValueError): verify_adapter_binding(preflight)


if __name__ == '__main__':
    unittest.main(verbosity=2)
