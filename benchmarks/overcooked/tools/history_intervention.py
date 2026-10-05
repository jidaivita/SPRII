"""Original fixed-weight external-history intervention functions.

Platform schedulers and original-run receipt locators are intentionally excluded.
All donor, support, inference and online environment mathematics are retained.
"""
import hashlib, json, os
from pathlib import Path
from contextlib import contextmanager
CONDITIONS = ('matched_external', 'null_external', 'wrong_external')
def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def verify(reference):
    require(sha(reference['path']) == reference['sha256'], 'Changed artifact: ' + reference['path'])


def write(path, value, *, replace=False):
    path = Path(path)
    require(replace or not path.exists(), 'Refusing overwrite: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def donor_index(partner_index):
    require(partner_index in (20, 21), 'Only the two frozen heldout policies are query partners')
    return 41 - partner_index


def pair_for_episode(episode):
    require(0 <= episode < 20, 'Evaluation episode outside the frozen chain')
    first = 2 * (episode // 5)
    return (first, first + 1)


def tensor_sha(params):
    # Same path/dtype/shape/bytes definition as bound native_a.train.tensor_sha.
    import numpy as np
    from flax import serialization
    h = hashlib.sha256()
    canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + [str(key)])
        else:
            a = np.asarray(value)
            h.update(canonical(path).encode()); h.update(str(a.dtype).encode())
            h.update(canonical(list(a.shape)).encode()); h.update(a.tobytes())
    visit(serialization.to_state_dict(params), [])
    return h.hexdigest()


def blank(length):
    import numpy as np
    return {'obs': np.zeros((length, 5, 5, 40), np.float32),
            'prev_actions': np.zeros(length, np.int32), 'prev_rewards': np.zeros(length, np.float32),
            'attention_mask': np.zeros(length, np.float32)}


def trace_tokens(trace):
    import numpy as np
    require(trace['obs_ego'].shape == (101, 5, 5, 40)
            and trace['actions'].shape == trace['rewards'].shape == (100, 2), 'Trace shape changed')
    out = blank(100)
    out['obs'][:] = trace['obs_ego'][:100]
    out['prev_actions'][1:] = trace['actions'][:-1, 0]
    out['prev_rewards'][1:] = trace['rewards'][:-1, 0]
    out['attention_mask'][:] = 1
    require(all(np.isfinite(v).all() for v in out.values()), 'Nonfinite public trace')
    require(((out['prev_actions'] >= 0) & (out['prev_actions'] < 6)).all(), 'Invalid ego action')
    return out


def support_bank(panel, indices):
    import numpy as np
    bank, used = {}, []
    for index in indices:
        for episode in range(8):
            rows = [r for r in panel['episodes'] if r['partner_index'] == index and r['episode_index'] == episode]
            require(len(rows) == 1, 'Missing or duplicate support episode')
            r = rows[0]
            require(r['probe_split'] == 'fit', 'Probe-test episodes cannot be support donors')
            verify({'path': r['npz_path'], 'sha256': r['sha256']})
            with np.load(r['npz_path'], allow_pickle=False) as trace:
                bank[index, episode] = trace_tokens(trace)
            used.append(r)
    require(len({r['fixed_ego_params_sha256'] for r in used}) == 1, 'Mixed donor collectors')
    return bank, used


def batch_from_query(query, bank, matched, wrong, pair):
    import numpy as np
    require(matched != wrong, 'Wrong donor must have another parameter identity')
    n = len(query)
    require(1 <= n <= 300 and len(pair) == len(set(pair)) == 2, 'Invalid query/pair budget')
    q = blank(300)
    for i, token in enumerate(query):
        for key in q:
            q[key][i] = 1 if key == 'attention_mask' else token[key]
    supports = [[bank[matched, e] for e in pair], [blank(100), blank(100)], [bank[wrong, e] for e in pair]]
    return {'query': {k: np.repeat(v[None], 3, axis=0) for k, v in q.items()},
            'support': {k: np.stack([np.stack([s[k] for s in two]) for two in supports]) for k in q}}, n


def inference(model, params, mode, backend):
    import jax
    import jax.numpy as jnp
    from native_a.model import apply_batch
    def forward(batch):
        if mode == 'upstream_ad':
            joined = {k: jnp.concatenate([batch['support'][k].reshape((3, 200) + v.shape[2:]), v], axis=1)
                      for k, v in batch['query'].items()}
            logits = model.apply(params, joined['obs'], joined['prev_actions'], joined['prev_rewards'],
                                 attention_mask=joined['attention_mask'], train=False)
            indices = 200 + batch['query']['attention_mask'].sum(-1).astype(jnp.int32) - 1
        else:
            logits = apply_batch(model, params, batch, train=False)['logits']
            indices = batch['query']['attention_mask'].sum(-1).astype(jnp.int32) - 1
        return logits[jnp.arange(3), indices]
    return jax.jit(forward, backend=backend)


def qualify_inference(cpu_infer, gpu_infer, bank):
    import numpy as np
    # Only training-partner fit traces qualify backend math, never heldout returns.
    query = []
    for e in (2, 3, 4):
        a = bank[0, e]
        query.extend([{k: v[i] for k, v in a.items() if k != 'attention_mask'} for i in range(100)])
    rows = []
    for length in (1, 17, 100, 300):
        batch, _ = batch_from_query(query[:length], bank, 0, 1, (0, 1))
        cpu, gpu = np.asarray(cpu_infer(batch)), np.asarray(gpu_infer(batch))
        require(np.isfinite(cpu).all() and np.isfinite(gpu).all(), 'Nonfinite qualification logits')
        np.testing.assert_allclose(cpu, gpu, atol=2e-5, rtol=2e-5)
        require(np.array_equal(cpu.argmax(-1), gpu.argmax(-1)), 'CPU/GPU qualification primitive actions differ')
        # Query equality means this is exactly a history intervention.
        require(all(np.array_equal(v[0], v[1]) and np.array_equal(v[0], v[2])
                    for v in batch['query'].values()), 'Conditions changed the query')
        rows.append({'query_length': length, 'max_abs_logit_error': float(abs(cpu-gpu).max()),
                     'argmax_equal': True})
    return {'status': 'PASS_CPU_GPU_SAME_WEIGHTS_ACTION_GATE', 'cases': rows,
            'atol': 2e-5, 'rtol': 2e-5, 'only_training_partner_fit_episodes': True}


@contextmanager
def host_only_partner_restore():
    """Read saved RL arrays on the CPU explicitly, independent of saved GPU ordinal.

    Keep native architecture/selection code; actual selected partner tensors must
    still match the pre-training isolation fingerprint immediately after loading.
    """
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    import common.save_load_utils as loaders
    import teammate_wrapper.rl_wrappers as wrappers
    original_load, original_wrapper = loaders.load_train_run, wrappers.load_train_run
    def host_read(path):
        path = Path(path)
        if not path.is_absolute():
            path = Path(loaders.REPO_PATH) / path
        cp = ocp.PyTreeCheckpointer()
        try:
            metadata = cp.metadata(str(path.resolve()))
            tree = metadata.tree if hasattr(metadata, 'tree') else metadata
            restore_args = jax.tree_util.tree_map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), tree)
            return cp.restore(str(path.resolve()), restore_args=restore_args)
        finally:
            cp.close()
    loaders.load_train_run = wrappers.load_train_run = host_read
    try:
        yield
    finally:
        loaders.load_train_run, wrappers.load_train_run = original_load, original_wrapper


def run_chain(infer, task, expected_partner_sha, bank, partner_index, condition, directory, callback):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from eval_icrl import create_env_for_task, create_teammate
    cpu = jax.devices('cpu')[0]
    with jax.default_device(cpu):
        env = create_env_for_task(task, max_steps=100)
        with host_only_partner_restore():
            teammate = create_teammate(task, env)
        require(tensor_sha(teammate._params) == expected_partner_sha, 'Fixed partner weights changed')
        previous_episodes, results, traces = [], [], []
        root_rng = jax.random.PRNGKey(62000)
        for episode in range(20):
            reset_rng, mate_rng, episode_rng = jax.random.split(jax.random.fold_in(root_rng, episode), 3)
            obs, state = env.reset(reset_rng)
            carry = teammate.init(mate_rng)
            done = {name: False for name in ('agent_0', 'agent_1', '__all__')}
            previous_action, previous_reward, total = 0, 0.0, 0.0
            current, episode_trace = [], []
            pair = pair_for_episode(episode)
            for step in range(100):
                current.append({'obs': np.asarray(obs['agent_0']), 'prev_actions': previous_action,
                                'prev_rewards': previous_reward})
                query = ([t for old in previous_episodes for t in old] + current)[-300:]
                batch, length = batch_from_query(query, bank, partner_index, donor_index(partner_index), pair)
                logits = np.asarray(infer(batch))
                require(logits.shape == (3, 6) and np.isfinite(logits).all(), 'Invalid ego logits')
                choices = logits.argmax(-1)
                action = int(choices[CONDITIONS.index(condition)])
                mate_key, step_key = jax.random.split(jax.random.fold_in(episode_rng, step), 2)
                available = env.get_avail_actions(state.env_state)['agent_1']
                carry, mate_action = teammate.act(carry, obs['agent_1'], jnp.array(done['agent_1']), mate_key,
                                                 env_state=state, avail_actions=available)
                if isinstance(mate_action, jax.Array):
                    require(all(d.platform == 'cpu' for d in mate_action.devices()), 'Partner inference moved from CPU')
                mate_action = int(np.asarray(mate_action).squeeze())
                obs, state, rewards, done, _ = env.step(step_key, state,
                                                      {'agent_0': action, 'agent_1': mate_action})
                previous_action, previous_reward = action, float(rewards['agent_0'])
                require(np.isfinite(previous_reward), 'Nonfinite native reward')
                if step == 0:
                    require(all(all(d.platform == 'cpu' for d in x.devices()) for x in jax.tree_util.tree_leaves(state)
                                if isinstance(x, jax.Array)), 'Environment computation moved away from CPU')
                total += previous_reward
                episode_trace.append(logits)
                if bool(done['__all__']):
                    break
            require(step == 99, 'Fixed native 100-step horizon changed')
            a = np.stack(episode_trace)
            a_prob = np.exp(a - a.max(-1, keepdims=True)); a_prob /= a_prob.sum(-1, keepdims=True)
            mean_prob = (a_prob[:, 0] + a_prob[:, 2]) / 2
            js = .5 * ((a_prob[:, 0] * np.log(np.maximum(a_prob[:, 0], 1e-30)/np.maximum(mean_prob, 1e-30))).sum(-1)
                       + (a_prob[:, 2] * np.log(np.maximum(a_prob[:, 2], 1e-30)/np.maximum(mean_prob, 1e-30))).sum(-1))
            row = {'episode': episode, 'return': total, 'steps': 100, 'condition': condition,
                   'partner_index': partner_index, 'task_id': task.task_id, 'support_pair': list(pair),
                   'wrong_donor_partner_index': donor_index(partner_index), 'query_length_at_end': length,
                   'matched_wrong_action_flip_fraction_fixed_query': float((a[:, 0].argmax(-1) != a[:, 2].argmax(-1)).mean()),
                   'matched_null_action_flip_fraction_fixed_query': float((a[:, 0].argmax(-1) != a[:, 1].argmax(-1)).mean()),
                   'matched_wrong_js_fixed_query': float(js.mean()), 'teammate_params_sha256': expected_partner_sha,
                   'counterfactual_query_distribution': condition, 'encoder_updates': 0}
            write(directory / ('episode_%02d.json' % episode), row)
            results.append(row); traces.append(a); callback(row)
            previous_episodes.append(current)
            previous_episodes = previous_episodes[-3:]
        np.savez_compressed(directory / 'fixed_query_logits.npz', logits=np.stack(traces),
                            condition_order=np.asarray(CONDITIONS))
        require(tensor_sha(teammate._params) == expected_partner_sha, 'Partner changed during rollout')
    return results, ref(directory / 'fixed_query_logits.npz')

