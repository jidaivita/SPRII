"""Collect a fixed CPU probe panel and common-input policy signatures, without training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_json(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(replace or not path.exists(), f'Committed artifact already exists: {path}')
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(f.name, path)


def atomic_npz(path, arrays, np):
    require(not path.exists(), f'Unreceipted or committed NPZ already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(f.name, path)


def configure(plan):
    require(plan['backend'] == 'cpu' and plan['threads'] == 8, 'This collector is the fixed eight-CPU worker')
    allowed = sorted(os.sched_getaffinity(0))
    require(len(allowed) >= 8, 'Eight allowed host CPUs are required')
    os.sched_setaffinity(0, allowed[:8])
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                      OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='8')


def check_episode(a, np):
    t = len(a['actions'])
    require(t == 100, 'Fixed panel requires a full 100-step independent episode')
    require(a['obs_ego'].shape == a['obs_partner'].shape == (t + 1, 5, 5, 40), 'Native observation shape changed')
    require(a['actions'].shape == a['rewards'].shape == a['done'].shape == (t, 2), 'Two-agent trace shape changed')
    require(a['done_all'].shape == (t,) and a['avail_actions'].shape == (t + 1, 2, 6), 'Done or action mask shape changed')
    require(all(np.isfinite(v).all() for v in a.values()), 'Nonfinite probe trace')
    require(np.issubdtype(a['actions'].dtype, np.integer) and ((a['actions'] >= 0) & (a['actions'] < 6)).all(), 'Invalid primitive action')
    require(not a['done_all'][:-1].any() and bool(a['done_all'][-1]), 'Unexpected native episode boundary')
    require(np.isin(a['avail_actions'], (0, 1)).all() and (a['avail_actions'].sum(-1) > 0).all(), 'Invalid action availability')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    configure(plan)
    require(plan['partner_count'] == 22 and plan['episodes_per_partner'] == 16 and plan['episode_horizon'] == 100,
            'Registered panel dimensions changed')
    require(plan['pairs_per_partner_per_split'] == 4 and plan['maximum_environment_steps'] == 35200, 'Panel budget changed')
    fit, test = plan['probe_fit_reset_seeds'], plan['probe_test_reset_seeds']
    require(len(fit) == len(test) == 8 and len(set(fit + test)) == 16, 'Fit/test reset seeds must be disjoint')
    require(plan['fixed_ego'] == {'source_seed': 4600, 'checkpoint_index': 4,
            'agent_id': 'agent_0', 'native_greedy_test_mode': True}, 'Fixed ego changed')
    repo, out = Path(plan['repo']).resolve(), Path(plan['out_dir']).resolve()
    sys.path.insert(0, str(repo))
    import fcntl
    import jax
    import jax.numpy as jnp
    import numpy as np
    from benchmarks.manifest_schema import TaskEntry
    from eval_icrl import create_env_for_task, create_teammate
    from teammate_wrapper.registry import make_teammate
    from teammate_wrapper.specs import RLTeammateSpec
    from native_a.train import tensor_sha
    require(jax.default_backend() == 'cpu', 'CPU fallback/selection mismatch')
    training, unseen = rows(plan['train_manifest']), rows(plan['unseen_manifest'])
    require(len(training) == 20 and len(unseen) == 2 and all(t['split'] == 'train' for t in training), 'Wrong training population')
    require(all(t['split'] in ('development', 'heldout') for t in unseen), 'Unseen external role must be preserved')
    tasks = training + unseen
    require(len({t['task_id'] for t in tasks}) == 22 and all(t['layout_name'] == plan['layout'] for t in tasks), 'Task identities/layout changed')
    isolation = read(plan['source_isolation'])
    require(isolation['status'] == 'SOURCE_AND_DATA_ISOLATION_PASS', 'Before-model-training source/data isolation has not passed')
    require(isolation['train_manifest']['sha256'] == digest(plan['train_manifest'])
            and isolation['unseen_manifest']['sha256'] == digest(plan['unseen_manifest']), 'Source manifests changed')
    for reference in isolation['data'].values():
        require(digest(reference['path']) == reference['sha256'], 'Model-training data binding changed')
    source_rows = {r['task_id']: r for r in isolation['sources']}
    require(set(source_rows) == {t['task_id'] for t in tasks}, 'Isolation source coverage differs')
    require(not out.exists() or args.resume, 'New panel requires a fresh directory; use --resume for the same frozen panel')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = ('envs/__init__.py', 'envs/overcooked_v2/overcooked.py', 'envs/overcooked_v2/overcooked_v2_wrapper.py',
             'eval_icrl.py', 'teammate_wrapper/registry.py', 'teammate_wrapper/rl_wrappers.py',
             'agents/cnn_rnn_actor_critic.py', 'agents/cnn_rnn_actor_critic_agent.py', 'native_a/train.py')
    binding = {'plan_sha256': digest(args.plan), 'collector_sha256': digest(__file__),
               'fixed_ego_task_sha256': digest(plan['fixed_ego_task']),
               'source_isolation': {'path': plan['source_isolation'], 'sha256': digest(plan['source_isolation'])},
               'source_sha256': {f: digest(repo / f) for f in files}, 'jax_version': jax.__version__,
               'backend': jax.default_backend(), 'zero_training_updates': True}
    if args.resume:
        require(read(out / 'binding.json') == binding, 'Frozen panel binding changed')
    else:
        atomic_json(out / 'binding.json', binding)
    env = create_env_for_task(TaskEntry.from_json(dict(tasks[0], split='train')), max_steps=100)
    from source_isolation import source_row
    fixed_task = read(plan['fixed_ego_task'])
    fixed_source = source_row(fixed_task, 'training')
    fixed = fixed_task['teammate']
    require(fixed_source['source_seed'] == 4600 and fixed['extra']['checkpoint_idx'] == 4,
            'Fixed collector must be original4600 checkpoint4')
    ego = make_teammate(RLTeammateSpec(algo='ippo', ckpt_path=fixed['ckpt'], use_log_wrapper=True, extra=fixed['extra']), env, 'agent_0')
    require(fixed['extra'].get('test_mode', True) is True, 'Fixed ego must keep native greedy test mode')
    ego_hash = tensor_sha(ego._params)
    require(ego_hash == plan['fixed_ego_parameter_sha256'], 'Fixed ego parameter identity changed')
    partners, policies = [], []
    for i, task in enumerate(tasks):
        native = TaskEntry.from_json(dict(task, split='train'))
        mate = create_teammate(native, env)
        identity = tensor_sha(mate._params)
        require(identity == source_rows[task['task_id']]['teammate_params_sha256'], 'Fixed partner parameter identity changed')
        partners.append({'partner_index': i, 'partner_identity_sha256': identity,
                         'partner_role': 'train' if i < 20 else 'heldout', 'source_task_id': task['task_id'],
                         'external_split': task['split'], 'source_seed': source_rows[task['task_id']]['source_seed'],
                         'checkpoint_index': task['teammate']['extra']['checkpoint_idx'], 'task_spec': task,
                         'native_uses_action_mask': bool(mate._policy.use_avail_actions)})
        policies.append(mate)
    require(len({r['partner_identity_sha256'] for r in partners}) == 22, 'Duplicate parameter identities')
    stop = None
    def request_stop(signum, frame):
        nonlocal stop
        stop = signum
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    episodes, started = [], time.monotonic()
    for i, (mate, partner) in enumerate(zip(policies, partners)):
        for ep, seed in enumerate(fit + test):
            npz = out / 'episodes' / f'partner_{i:02d}_episode_{ep:02d}.npz'
            receipt = npz.with_suffix('.json')
            metadata = {k: partner[k] for k in ('partner_index', 'partner_identity_sha256', 'partner_role', 'source_task_id')}
            metadata.update(episode_index=ep, probe_split='fit' if ep < 8 else 'test', pair_index=(ep % 8) // 2,
                            reset_seed=seed, npz_path=str(npz), fixed_ego_params_sha256=ego_hash)
            if receipt.exists():
                row = read(receipt)
                require(all(row[k] == v for k, v in metadata.items()) and row['sha256'] == digest(npz), 'Episode receipt changed')
                with np.load(npz, allow_pickle=False) as stored:
                    check_episode(dict(stored), np)
                episodes.append(row)
                continue
            if stop:
                atomic_json(out / 'progress.json', {'status': 'PARTIAL', 'completed_episodes': len(episodes), 'signal': stop}, True)
                return 128 + stop
            reset_key, ego_key, mate_key, step_base = jax.random.split(jax.random.PRNGKey(seed), 4)
            obs, state = env.reset(reset_key)
            ego_carry, mate_carry = ego.init(ego_key), mate.init(mate_key)
            previous_done = {'agent_0': False, 'agent_1': False}
            trace = {'obs_ego': [np.asarray(obs['agent_0'])], 'obs_partner': [np.asarray(obs['agent_1'])],
                     'actions': [], 'rewards': [], 'done': [], 'done_all': [], 'avail_actions': []}
            avail = env.get_avail_actions(state.env_state)
            trace['avail_actions'].append(np.stack([np.asarray(avail[a]) for a in ('agent_0', 'agent_1')]))
            for step in range(100):
                ego_rng, mate_rng, env_rng = jax.random.split(jax.random.fold_in(step_base, step), 3)
                ego_carry, ea = ego.act(ego_carry, obs['agent_0'], jnp.asarray(previous_done['agent_0']), ego_rng, env_state=state, avail_actions=avail['agent_0'])
                mate_carry, ma = mate.act(mate_carry, obs['agent_1'], jnp.asarray(previous_done['agent_1']), mate_rng, env_state=state, avail_actions=avail['agent_1'])
                actions = [int(np.asarray(ea).squeeze()), int(np.asarray(ma).squeeze())]
                require(all(0 <= a < 6 for a in actions), 'Policy returned invalid primitive action')
                for agent, wrapper in enumerate((ego, mate)):
                    if wrapper._policy.use_avail_actions:
                        require(bool(np.asarray(avail[f'agent_{agent}'])[actions[agent]]), 'Native masked action is unavailable')
                obs, state, reward, done, info = env.step(env_rng, state, dict(zip(('agent_0', 'agent_1'), actions)))
                trace['actions'].append(actions)
                trace['rewards'].append([float(reward[a]) for a in ('agent_0', 'agent_1')])
                trace['done'].append([bool(done[a]) for a in ('agent_0', 'agent_1')]); trace['done_all'].append(bool(done['__all__']))
                trace['obs_ego'].append(np.asarray(obs['agent_0'])); trace['obs_partner'].append(np.asarray(obs['agent_1']))
                avail = env.get_avail_actions(state.env_state)
                trace['avail_actions'].append(np.stack([np.asarray(avail[a]) for a in ('agent_0', 'agent_1')]))
                previous_done = done
                if bool(done['__all__']):
                    break
            arrays = {k: np.asarray(v, dtype=np.int32 if k == 'actions' else np.bool_ if k in ('done', 'done_all') else np.float32) for k, v in trace.items()}
            check_episode(arrays, np)
            require(tensor_sha(ego._params) == ego_hash and tensor_sha(mate._params) == partner['partner_identity_sha256'], 'Policy parameters changed during read-only rollout')
            atomic_npz(npz, arrays, np)
            metadata.update(sha256=digest(npz), steps=len(arrays['actions']), ego_return=float(arrays['rewards'][:, 0].sum()),
                            partner_return=float(arrays['rewards'][:, 1].sum()), reset_key=np.asarray(reset_key).tolist(),
                            step_base_key=np.asarray(step_base).tolist())
            atomic_json(receipt, metadata)
            episodes.append(metadata)
            progress = {'status': 'COLLECTING', 'completed_episodes': len(episodes), 'maximum_episodes': 352,
                        'environment_steps': sum(e['steps'] for e in episodes), 'elapsed_seconds': time.monotonic() - started}
            atomic_json(out / 'progress.json', progress, True)
            print(json.dumps(progress), flush=True)
    require(len(episodes) == 352 and sum(e['steps'] for e in episodes) == 35200, 'Panel budget incomplete')
    panel = {'schema': 'native-probe-panel/1', 'status': 'PASS', 'episodes': episodes, 'partners': partners,
             'source_isolation': binding['source_isolation'], 'fixed_ego': {'source_task_id': fixed_task['task_id'], 'params_sha256': ego_hash},
             'binding_sha256': digest(out / 'binding.json'), 'episode_horizon': 100, 'zero_training_updates': True,
             'pairing': plan['pairing'], 'private_partner_observations_are_encoder_inputs': False,
             'fit_reset_seeds': fit, 'test_reset_seeds': test}
    manifest = out / 'panel_manifest.json'
    if manifest.exists():
        require(read(manifest) == panel, 'Completed panel manifest changed')
    else:
        atomic_json(manifest, panel)
    signature_plan = plan['common_input_signature']
    require(signature_plan['training_partner_indices'] == list(range(20)) and signature_plan['episode_index'] == 0
            and signature_plan['snippet_starts'] == [0, 32] and signature_plan['snippet_length'] == 8, 'Common input bank changed')
    signature_path, signature_manifest = out / 'signature.npz', out / 'signature_manifest.json'
    if signature_manifest.exists():
        old = read(signature_manifest)
        require(old['panel_manifest']['sha256'] == digest(manifest) and old['npz']['sha256'] == digest(signature_path), 'Signature binding changed')
        return 0
    anchors, masks, sources = [], [], []
    for i in range(20):
        row = next(r for r in episodes if r['partner_index'] == i and r['episode_index'] == 0)
        require(row['probe_split'] == 'fit' and row['partner_role'] == 'train', 'Anchor source must be a training-partner fit episode')
        with np.load(row['npz_path'], allow_pickle=False) as trace:
            for start in (0, 32):
                anchors.append(trace['obs_partner'][start:start + 8]); masks.append(trace['avail_actions'][start:start + 8, 1])
                sources.append({'partner_identity_sha256': row['partner_identity_sha256'], 'episode_index': 0,
                                'episode_npz': {'path': row['npz_path'], 'sha256': row['sha256']}, 'start': start, 'length': 8})
    anchor_obs, anchor_mask = np.stack(anchors), np.stack(masks)
    anchor_done = np.zeros((40, 8), np.bool_); anchor_done[:, 0] = True
    probabilities = []
    for i, mate in enumerate(policies):
        hidden = mate._policy.init_hstate(batch_size=40, aux_info={'agent_id': 1})
        require(bool(np.all(np.asarray(hidden) == 0)), 'Signature must start from zero GRU carry')
        _, _, distribution, _ = mate._policy.get_action_value_policy(mate._params,
            jnp.asarray(anchor_obs.swapaxes(0, 1)), jnp.asarray(anchor_done.swapaxes(0, 1)),
            jnp.asarray(anchor_mask.swapaxes(0, 1)), hidden, jax.random.PRNGKey(54000 + i))
        prob = np.asarray(distribution.probs).swapaxes(0, 1)
        require(prob.shape == (40, 8, 6) and np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all()
                and np.allclose(prob.sum(-1), 1, atol=1e-5), 'Invalid common-input action probabilities')
        if mate._policy.use_avail_actions:
            require((prob[anchor_mask == 0] <= 1e-6).all(), 'Signature ignored native action masking')
        require(tensor_sha(mate._params) == partners[i]['partner_identity_sha256'], 'Signature computation changed policy parameters')
        probabilities.append(prob)
    probabilities = np.stack(probabilities)
    atomic_npz(signature_path, {'probabilities': probabilities, 'greedy_actions': probabilities.argmax(-1).astype(np.int32),
                              'anchor_obs': anchor_obs, 'anchor_avail_actions': anchor_mask, 'anchor_done': anchor_done}, np)
    signature = {'schema': 'native-policy-function-signature/1', 'status': 'PASS',
                 'panel_manifest': {'path': str(manifest), 'sha256': digest(manifest)},
                 'npz': {'path': str(signature_path), 'sha256': digest(signature_path)},
                 'partner_identity_sha256s': [p['partner_identity_sha256'] for p in partners],
                 'probabilities_shape': [22, 40, 8, 6], 'flat_dimensions': 1920, 'anchors': sources,
                 'hidden_state': 'Independent zero GRU at each common snippet',
                 'claim': 'Fixed-policy action-distribution signature on a shared input bank, not alpha or weight recovery',
                 'zero_training_updates': True}
    atomic_json(signature_manifest, signature)
    atomic_json(out / 'progress.json', {'status': 'PASS', 'completed_episodes': 352, 'environment_steps': 35200,
                'panel_manifest': str(manifest), 'signature_manifest': str(signature_manifest), 'zero_training_updates': True}, True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
