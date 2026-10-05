"""CPU inference adapters for the unchanged native teamwork evaluation.

Right padding plus an explicit key mask gives the same valid-prefix logits as
the original variable-length AD. It avoids compiling a different CNN and
attention shape on each interaction step. This is an execution optimization,
not a change in observation or action permissions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
_threads = 4
for _i, _arg in enumerate(sys.argv):
    if _arg == '--threads' and _i + 1 < len(sys.argv):
        _threads = int(sys.argv[_i + 1])
    elif _arg.startswith('--threads='):
        _threads = int(_arg.split('=', 1)[1])
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = str(_threads)

import jax
import jax.numpy as jnp
import numpy as np


class StaticADInference:
    def __init__(self, model, max_len=500):
        self.model = model
        self.max_len = max_len
        self.use_teammate_actions = model.config.use_teammate_actions

        @jax.jit
        def infer(params, obs, actions, rewards, mask, teammate_actions):
            return model.apply(
                params, obs, actions, rewards, attention_mask=mask,
                prev_teammate_actions=teammate_actions if self.use_teammate_actions else None,
                train=False)

        self._infer = infer

    def apply(self, params, obs, prev_actions, prev_rewards,
              attention_mask=None, prev_teammate_actions=None, train=False, **kwargs):
        if train or kwargs:
            raise ValueError('StaticADInference is an evaluation-only adapter')
        obs, actions, rewards = map(np.asarray, (obs, prev_actions, prev_rewards))
        batch, length = actions.shape
        if not 0 < length <= self.max_len:
            raise ValueError((length, self.max_len))
        if tuple(obs.shape[:2]) != (batch, length) or rewards.shape != actions.shape:
            raise ValueError('Observation/action/reward prefix dimensions differ')
        if self.use_teammate_actions and prev_teammate_actions is None:
            raise ValueError('Configured past teammate actions are missing')
        if not self.use_teammate_actions and prev_teammate_actions is not None:
            raise ValueError('Past teammate actions were not granted to this model')
        def padded(value):
            out = np.zeros((batch, self.max_len, *value.shape[2:]), dtype=value.dtype)
            out[:, :length] = value
            return out
        mask = np.zeros((batch, self.max_len), dtype=np.float32)
        mask[:, :length] = 1 if attention_mask is None else np.asarray(attention_mask)
        mate = np.zeros_like(actions) if prev_teammate_actions is None else np.asarray(prev_teammate_actions)
        logits = self._infer(params, padded(obs), padded(actions), padded(rewards), mask, padded(mate))
        # Slice on the host so changing prefix lengths do not compile XLA slices.
        return np.asarray(logits)[:, :length]


def check_static_equivalence(model, params, obs_shape, max_len=500):
    """Check the only numerical change against the upstream inference path."""
    adapter = StaticADInference(model, max_len)
    rng = np.random.default_rng(9124100)
    receipts = []
    for length in (1, 3, 17):
        obs = rng.integers(0, 2, (1, length, *obs_shape)).astype(np.float32)
        actions = rng.integers(0, model.config.num_actions, (1, length), dtype=np.int32)
        rewards = rng.normal(size=(1, length)).astype(np.float32)
        mate = rng.integers(0, model.config.num_actions, (1, length), dtype=np.int32) if adapter.use_teammate_actions else None
        native = model.apply(params, jnp.asarray(obs), jnp.asarray(actions), jnp.asarray(rewards),
                             prev_teammate_actions=mate, train=False)
        static = adapter.apply(params, obs, actions, rewards, prev_teammate_actions=mate)
        np.testing.assert_allclose(static, np.asarray(native), atol=2e-5, rtol=2e-5)
        receipts.append({'prefix_length': length, 'max_abs_error': float(np.max(np.abs(static-np.asarray(native))))})
    return receipts


def evaluate_native_task(model, params, task_entry, *, mode, episodes=10,
                         max_steps=100, seed=4200, query_len=300,
                         support_len=100, history_condition='matched', episode_callback=None,
                         expected_teammate_params_sha256=None):
    """Evaluate paired models with identical past-only observation budgets.

    Uses upstream environment/teammate creation and reward semantics.  The
    query is the most recent 300 tokens; supports are the two most recent
    completed episodes strictly before the first episode touched by the query.
    Missing early supports are explicitly masked. No privileged partner IDs or
    current/future teammate actions enter the policy. Null history is an
    evaluation intervention, not a separately enhanced controller.
    """
    if jax.default_backend() != 'cpu':
        raise RuntimeError('Native evaluation must run entirely on CPU')
    if task_entry.split != 'train' or task_entry.teammate.kind != 'rl':
        raise ValueError('This development evaluator accepts only custom train-split RL tasks')
    if mode not in ('baseline', 'none', 'VC', 'I+VC'):
        raise ValueError(mode)
    if history_condition not in ('matched', 'null'):
        raise ValueError(history_condition)
    if episodes < 1 or max_steps != 100 or support_len != 100 or query_len != 300:
        raise ValueError('This paired protocol uses official 100-step episodes and 2x100+300 tokens')
    from eval_icrl import create_env_for_task, create_teammate
    from native_a.model import apply_batch
    ad_config = model.config if mode == 'baseline' else model.config.ad
    shape = tuple(ad_config.obs_shape)
    env = create_env_for_task(task_entry, max_steps=max_steps)
    teammate = create_teammate(task_entry, env)
    from native_a.train import tensor_sha
    if teammate._params is None:
        raise ValueError('The development RL teammate has no loaded parameter tensor')
    teammate_sha = tensor_sha(teammate._params)
    if expected_teammate_params_sha256 is not None and teammate_sha != expected_teammate_params_sha256:
        raise ValueError('RL partner weights changed between paired history conditions')

    if mode == 'baseline':
        @jax.jit
        def infer(batch):
            joined = baseline_batch_jax(batch)
            logits = model.apply({'params': params}, **joined, train=False)
            n = batch['query']['attention_mask'].sum(axis=1).astype(jnp.int32)
            return logits[jnp.arange(n.shape[0]), 2 * support_len + n - 1]
    else:
        @jax.jit
        def infer(batch):
            out = apply_batch(model, params, batch, train=False)
            n = batch['query']['attention_mask'].sum(axis=1).astype(jnp.int32)
            return out['logits'][jnp.arange(n.shape[0]), n - 1]

    def blank(length):
        data = {'obs': np.zeros((1, length, *shape), np.float32),
                'prev_actions': np.zeros((1, length), np.int32),
                'prev_rewards': np.zeros((1, length), np.float32),
                'attention_mask': np.zeros((1, length), np.float32)}
        if ad_config.use_teammate_actions:
            data['prev_teammate_actions'] = np.zeros((1, length), np.int32)
        return data

    def fill(tokens, length):
        data = blank(length)
        for j, token in enumerate(tokens[-length:]):
            for key in data:
                data[key][0, j] = 1 if key == 'attention_mask' else token[key]
        return data

    rng = jax.random.PRNGKey(seed)
    completed, current, results = [], [], []
    for ep in range(episodes):
        # Same exogenous reset/noise seeds across matched/null and methods even
        # when a previous rollout terminated at a different time.
        reset_rng, mate_init_rng, episode_rng = jax.random.split(jax.random.fold_in(rng, ep), 3)
        obs, state = env.reset(reset_rng)
        if tuple(np.asarray(obs['agent_0']).shape) != shape:
            raise ValueError('Native environment observation shape differs from checkpoint')
        carry = teammate.init(mate_init_rng)
        done = {name: False for name in ('agent_0', 'agent_1', '__all__')}
        # Preserve the upstream online AD reset convention.  The offline
        # upstream HDF5 adapter instead retains predecessor values across done;
        # this existing training/evaluation convention difference is recorded.
        previous_action, previous_reward, previous_mate = 0, 0.0, 0
        current, total_reward = [], 0.0
        context_stats = []
        for step in range(max_steps):
            token = {'obs': np.asarray(obs['agent_0']), 'prev_actions': previous_action,
                     'prev_rewards': previous_reward, 'episode': ep}
            if ad_config.use_teammate_actions:
                token['prev_teammate_actions'] = previous_mate
            current.append(token)
            recent = [t for episode in completed for t in episode] + current
            query = recent[-query_len:]
            query_first_ep = query[0]['episode']
            supports = [episode for episode in completed if episode[-1]['episode'] < query_first_ep][-2:]
            if any(s[-1]['episode'] >= query_first_ep for s in supports):
                raise AssertionError('A support episode reaches into the query')
            available_support_ids = [s[-1]['episode'] for s in supports]
            if history_condition == 'null':
                supports = []
            support_arrays = [fill(s, support_len) for s in supports]
            support_arrays = [blank(support_len)] * (2-len(support_arrays)) + support_arrays
            batch = {'query': fill(query, query_len),
                     'support': {k: np.stack([s[k][0] for s in support_arrays], axis=0)[None]
                                 for k in support_arrays[0]}}
            current_logits = np.asarray(infer(batch))
            if not np.isfinite(current_logits).all():
                raise FloatingPointError('Nonfinite ego-action logits during native evaluation')
            action = int(np.argmax(current_logits[0]))
            if not 0 <= action < ad_config.num_actions:
                raise ValueError('Policy produced an invalid native ego action')
            mate_rng, step_rng = jax.random.split(jax.random.fold_in(episode_rng, step), 2)
            available = env.get_avail_actions(state.env_state)['agent_1']
            carry, mate_action = teammate.act(carry, obs['agent_1'], jnp.array(done['agent_1']), mate_rng,
                                              env_state=state, avail_actions=available)
            mate_action = int(np.asarray(mate_action).squeeze())
            obs, state, rewards, done, info = env.step(step_rng, state,
                {'agent_0': action, 'agent_1': mate_action})
            previous_action, previous_reward, previous_mate = action, float(rewards['agent_0']), mate_action
            if not np.isfinite(previous_reward):
                raise ValueError('Native environment returned a nonfinite reward')
            total_reward += previous_reward
            context_stats.append(len(query) + sum(len(s) for s in supports))
            if context_stats[-1] > 500:
                raise AssertionError('Observed history exceeds the registered 500-token budget')
            if bool(done['__all__']):
                break
        results.append({'episode': ep, 'return': total_reward, 'steps': step+1,
                        'max_observed_tokens': max(context_stats),
                        'available_support_episodes': len(available_support_ids),
                        'consumed_support_episodes': len(supports),
                        'final_query_first_episode': query_first_ep,
                        'final_support_episode_ids': [s[-1]['episode'] for s in supports],
                        'teammate_params_sha256': teammate_sha,
                        'success': total_reward > 0})
        if episode_callback is not None:
            episode_callback(dict(results[-1]))
        print({'native_eval_episode': results[-1], 'mode': mode, 'history': history_condition}, flush=True)
        completed.append(current)
        # Keep enough complete episodes for a full recent query plus two
        # strictly earlier supports, including episodes shorter than 100 steps.
        while len(completed) > 3 and sum(len(s) for s in completed[3:]) >= query_len:
            completed.pop(0)
    values = np.asarray([r['return'] for r in results])
    return {'task_id': task_entry.task_id, 'mode': mode, 'history_condition': history_condition,
            'teammate_params_sha256': teammate_sha,
            'seed': seed, 'episodes': results, 'mean_return': float(values.mean()),
            'auc': float(values.sum()), 'context_budget': 500, 'max_steps': max_steps,
            'history_intervention_scope': 'support_only; the dynamic query keeps its recent episode context',
            'previous_token_at_episode_reset': 'zero, as in upstream evaluate_ad_task',
            'offline_upstream_reset_difference': 'training HDF5 query windows retain predecessor tokens across episode boundaries',
            'rng_rule': 'fold_in(base_seed, episode), then fold_in(episode_rng, step)',
            'evaluation_support_rule': 'two most recent completed episodes strictly before the first query episode',
            'success_rate': float(np.mean([r['success'] for r in results])),
            'test_read': False, 'development_only': True,
            'greedy': True, 'upstream_environment_and_teammate': True,
            'scope': 'native teamwork evaluation; aggregation unit is partner, not timestep'}


def baseline_batch_jax(batch):
    query, supports = batch['query'], batch['support']
    out = {}
    for key in ('obs', 'prev_actions', 'prev_rewards', 'attention_mask', 'prev_teammate_actions'):
        if key in query:
            s = supports[key]
            s = s.reshape((s.shape[0], s.shape[1]*s.shape[2], *s.shape[3:]))
            out[key] = jnp.concatenate([s, query[key]], axis=1)
    return out


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_development_tasks(manifest, selected_task_ids=None):
    """Read only a custom train-split RL manifest, never official test tasks."""
    from benchmarks.manifest_schema import TaskEntry
    path = Path(manifest).resolve()
    official = Path(__file__).resolve().parents[1] / 'benchmarks' / 'overcooked_icrl'
    if path.is_relative_to(official.resolve()):
        raise ValueError('Official benchmark manifests cannot be used for native-A development')
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(row.get('split') != 'train' or row.get('teammate', {}).get('kind') != 'rl' for row in rows):
        raise ValueError('Every entry must be a custom train-split RL task; test access is forbidden')
    tasks = [TaskEntry.from_json(row) for row in rows]
    ids = [task.task_id for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate task IDs in the development manifest')
    if selected_task_ids:
        if len(set(selected_task_ids)) != len(selected_task_ids) or not set(selected_task_ids).issubset(ids):
            raise ValueError('Selected task IDs must exist exactly once in the manifest')
        selected = set(selected_task_ids)
        tasks = [task for task in tasks if task.task_id in selected]
    return tasks


def main():
    parser = argparse.ArgumentParser(description='CPU development teamwork returns with a frozen native-AD checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Native checkpoint, pointer JSON, or run directory')
    parser.add_argument('--manifest', required=True, help='Custom train-split RL task JSONL; official test forbidden')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--task-id', action='append', help='Optional fixed task subset; may be repeated')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=4200)
    parser.add_argument('--history-condition', choices=('matched', 'null', 'both'), default='both')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 24 or args.episodes < 1:
        raise ValueError('Invalid CPU thread count or episode budget')

    from native_a.train import load_native_checkpoint, atomic_json, cpu_receipt
    tasks = load_development_tasks(args.manifest, args.task_id)
    model, params, config, checkpoint = load_native_checkpoint(args.checkpoint)
    budget = config['sampler']
    expected = {'query_len': 300, 'support_count': 2, 'support_len': 100, 'total_history_tokens': 500}
    if any(budget.get(key) != value for key, value in expected.items()):
        raise ValueError('Checkpoint was not trained with the frozen 2x100+300 token contract')
    root = Path(__file__).resolve().parents[1]
    for name in ('native_a/model.py', 'benchmarks/baselines/ad/model.py'):
        if checkpoint['code_sha256'].get(name) != _digest(root / name):
            raise ValueError(f'Inference model source differs from checkpoint: {name}')
    output = Path(args.out_dir).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite or silently resume a development evaluation')
    output.mkdir(parents=True)
    manifest_hash = _digest(args.manifest)
    conditions = ['matched', 'null'] if args.history_condition == 'both' else [args.history_condition]
    evaluation_id = hashlib.sha256(json.dumps({
        'checkpoint': checkpoint['sha256'], 'manifest': manifest_hash,
        'tasks': [t.task_id for t in tasks], 'seed': args.seed,
        'episodes': args.episodes, 'conditions': conditions,
    }, sort_keys=True).encode()).hexdigest()
    receipt = {
        'format': 'native-ad-a/evaluation/1', 'status': 'RUNNING', 'evaluation_id': evaluation_id,
        'mode': config['mode'], 'checkpoint': checkpoint,
        'manifest': {'path': str(Path(args.manifest).resolve()), 'sha256': manifest_hash},
        'task_ids': [t.task_id for t in tasks], 'conditions': conditions,
        'episodes_per_task_condition': args.episodes, 'seed': args.seed,
        'query_len': 300, 'support_count': 2, 'support_len': 100, 'max_steps': 100,
        'development_only': True, 'test_read': False, 'optimizer_updates': 0,
        'history_intervention_scope': 'support_only',
        'checkpoint_step': checkpoint['step'],
        'checkpoint_is_terminal': checkpoint['step'] == config['num_steps'],
        'code_sha256': {name: _digest(root / name) for name in (
            'native_a/evaluate.py', 'native_a/model.py', 'eval_icrl.py',
            'benchmarks/baselines/ad/model.py', 'envs/__init__.py')},
        'cpu': cpu_receipt(), 'completed_episode_files': [], 'task_results': [],
    }
    atomic_json(output / 'evaluation.json', receipt)
    started = time.monotonic()
    try:
        for task in tasks:
            task_key = hashlib.sha256(task.task_id.encode()).hexdigest()[:16]
            task_dir = output / 'tasks' / task_key
            atomic_json(task_dir / 'task.json', task.to_json())
            expected_teammate_sha = None
            for condition in conditions:
                directory = task_dir / condition

                def on_episode(row):
                    path = directory / f"episode_{row['episode']:04d}.json"
                    if path.exists():
                        raise FileExistsError(f'Committed episode already exists: {path}')
                    entry = {**row, 'evaluation_id': evaluation_id, 'task_id': task.task_id,
                             'mode': config['mode'], 'history_condition': condition, 'seed': args.seed,
                             'checkpoint_sha256': checkpoint['sha256'], 'manifest_sha256': manifest_hash,
                             'test_read': False}
                    atomic_json(path, entry)
                    receipt['completed_episode_files'].append({
                        'path': str(path.relative_to(output)), 'sha256': _digest(path)})
                    receipt['elapsed_seconds'] = time.monotonic() - started
                    atomic_json(output / 'evaluation.json', receipt)

                result = evaluate_native_task(
                    model, params, task, mode=config['mode'], episodes=args.episodes,
                    seed=args.seed, history_condition=condition, episode_callback=on_episode,
                    expected_teammate_params_sha256=expected_teammate_sha)
                expected_teammate_sha = result['teammate_params_sha256']
                atomic_json(directory / 'result.json', result)
                receipt['task_results'].append({
                    'task_id': task.task_id, 'history_condition': condition,
                    'mean_return': result['mean_return'], 'auc': result['auc'],
                    'path': str((directory / 'result.json').relative_to(output)),
                    'sha256': _digest(directory / 'result.json')})
        if _digest(args.manifest) != manifest_hash:
            raise RuntimeError('Manifest changed during evaluation')
        if any(_digest(root / name) != digest for name, digest in receipt['code_sha256'].items()):
            raise RuntimeError('Evaluation source changed during the run')
        receipt['status'] = 'PASS_EXECUTION'
        receipt['claim'] = 'Development return measurements; execution success is not a scientific improvement claim'
    except BaseException as error:
        receipt['status'] = 'FAILED'
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        atomic_json(output / 'evaluation.json', receipt)
    print(json.dumps({'status': receipt['status'], 'episodes_saved': len(receipt['completed_episode_files']),
                      'result': str(output / 'evaluation.json')}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
