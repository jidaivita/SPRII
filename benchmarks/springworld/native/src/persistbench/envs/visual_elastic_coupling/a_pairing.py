"""Private, balanced nine-arm A training plans; no image or test-data reads.

One sweep presents every training system once on each branch. Regular relation
graphs are decomposed into perfect matchings, sampled uniformly: all admissible
edges then have exactly 1/degree probability, with exact donor marginals. This
does not claim independent pairs within a sweep or uniformly sampled matchings.
"""
import copy
import hashlib
import json
import numpy as np
from .pairing import WindowSupport, validate_pair, validate_relation

CONFIGURATIONS = {
    'Supervised-Split': ('B0_split', 'G3', 'Independent', 0., 0.),
    'B0': ('B0', 'G3', 'Independent', 0., 0.),
    'Split': ('B0_split', 'G3', 'Independent', 0., 0.),
    'Align': ('B2', 'G3', 'Independent', 1., 0.),
    'Cross': ('Bx', 'G3', 'Independent', 0., .1),
    'Both': ('B3', 'G3', 'Independent', 1., .1),
    'Both-SameEp': ('B3', 'G3', 'SameEp', 1., .1),
    'Both-Random': ('B3', 'Random', 'Independent', 1., .1),
    'Both-G1': ('B3', 'G1', 'Independent', 1., .1),
    'Both-G2': ('B3', 'G2', 'Independent', 1., .1),
}
SCHEMA = 'vec.A-paired-training.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def configuration(name):
    if name not in CONFIGURATIONS: raise ValueError('unregistered A training configuration')
    variant, relation, profile, alignment, cross = CONFIGURATIONS[name]
    return dict(name=name, variant=variant, relation=relation, pairing_profile=profile,
        lambda_p=alignment, lambda_x=cross, sigreg_weight=.02, cross_direction='donor_to_recipient',
        self_branches='both', horizons=[1, 4, 16], normalization='train_history_running_statistics_pre_step_v1')


def _matchings(adjacency, rng):
    """Edge-disjoint perfect matchings of a regular bipartite relation graph."""
    n = len(adjacency); degree = len(adjacency[0])
    if degree < 1 or any(len(row) != degree or len(set(row)) != degree for row in adjacency):
        raise ValueError('relation candidates require a nonempty regular graph')
    incoming = np.bincount([j for row in adjacency for j in row], minlength=n)
    if incoming.shape != (n,) or not np.all(incoming == degree):
        raise ValueError('donor relation marginals are not regular; register another balanced sampler')
    remaining = [set(row) for row in adjacency]; result = []
    for _ in range(degree):
        owners = [-1] * n
        neighbors = [rng.permutation(sorted(row)).tolist() for row in remaining]
        def assign(i, seen):
            for j in neighbors[i]:
                if j in seen: continue
                seen.add(j)
                if owners[j] < 0 or assign(owners[j], seen):
                    owners[j] = i; return True
            return False
        for i in rng.permutation(n).tolist():
            if not assign(i, set()): raise ValueError('regular relation graph failed perfect matching')
        mapping = [-1] * n
        for j, i in enumerate(owners): mapping[i] = j; remaining[i].remove(j)
        result.append(mapping)
    if any(remaining): raise ValueError('relation decomposition lost candidate edges')
    return result


class APairSchedule:
    """Candidate training-only schedule; formal acceptance is external.

    Only a declared episode kind is included (default forced). Physical failures
    are not removed: planned support is based on requested frames, and a missing
    actual prefix stops batch assembly without replacement. Other kinds are not
    silently mixed into the episode/action distribution.
    """
    def __init__(self, manifest, *, seed, kind='forced', history_frames=24, pairs_per_batch=48):
        if type(seed) is not int or not 0 <= seed < 2**32: raise ValueError('registered uint32 sampling seed required')
        if history_frames not in (24, 48, 96) or type(history_frames) is not int:
            raise ValueError('unregistered A history budget')
        if type(pairs_per_batch) is not int or pairs_per_batch < 2: raise ValueError('at least two pairs per branch required')
        if kind != 'forced': raise ValueError('this candidate sampler declares forced episodes; other kinds need a new profile')
        self.seed = seed; self.kind = kind; self.length = history_frames; self.batch_pairs = pairs_per_batch
        rows = [copy.deepcopy(r) for r in manifest['episodes'] if r['split'] == 'train' and r['kind'] == kind]
        if not rows: raise ValueError('no registered training episodes')
        self.rows = {}; self.systems = {}; self.episodes = {}
        for row in rows:
            key = row['episode_key']; system = row['system_key']; theta = row['theta']
            if key in self.rows: raise ValueError('duplicate training episode')
            if len(theta) != 3 or not np.isfinite(theta).all() or min(theta) <= 0:
                raise ValueError('invalid physical relation tuple')
            if system in self.systems and self.systems[system] != theta: raise ValueError('system changes physical tuple')
            self.systems[system] = theta
            self.rows[key] = row; self.episodes.setdefault(system, []).append(key)
        self.keys = sorted(self.systems); n = len(self.keys)
        if len({tuple(v) for v in self.systems.values()}) != n: raise ValueError('duplicate physical systems')
        if n % pairs_per_batch: raise ValueError('complete system sweeps must divide into whole pair batches without dropping systems')
        counts = {len(v) for v in self.episodes.values()}
        if len(counts) != 1 or min(counts) < 2: raise ValueError('equal independent episode counts per system required')
        self.episode_count = counts.pop()
        for system, keys in self.episodes.items():
            ordered = sorted(keys, key=lambda key: self.rows[key]['replicate'])
            if [self.rows[k]['replicate'] for k in ordered] != list(range(self.episode_count)):
                raise ValueError('episode replicate positions must be complete and unique')
            self.episodes[system] = ordered
        requested = {r['requested_frames'] for r in rows}
        if len(requested) != 1: raise ValueError('common requested duration required for matched temporal templates')
        self.requested_frames = requested.pop()
        if type(self.requested_frames) is not int or self.requested_frames < 2*self.length+16:
            raise ValueError('two disjoint A histories and complete future targets do not fit')
        self.windows = [(d, r) for r in range(self.length, self.requested_frames-self.length-15)
                        for d in range(r-self.length+1)]
        self.decompositions = {}; self.candidates = {}
        theta = np.asarray([self.systems[k] for k in self.keys])
        for relation, shared, varied in (('G1', (0,), (1, 2)), ('G2', (0, 1), (2,))):
            adjacency = [[j for j in range(n) if all(theta[i, a] == theta[j, a] for a in shared)
                          and all(theta[i, a] != theta[j, a] for a in varied)] for i in range(n)]
            self.candidates[relation] = adjacency
            self.decompositions[relation] = _matchings(adjacency, self._rng(0, 11 if relation == 'G1' else 12))
        self.decomposition_sha256 = digest(self.decompositions)
        self.manifest_sha256 = digest(manifest)
        self.selected_manifest_sha256 = digest([self.rows[k] for k in sorted(self.rows)])
        self._settings = digest([self.seed,self.kind,self.length,self.batch_pairs,self.keys,self.systems,
                                 self.episodes,self.episode_count,self.requested_frames,self.windows])

    def _rng(self, step, stream): return np.random.default_rng(np.random.SeedSequence([self.seed, step, stream]))

    def _guard(self):
        if (digest([self.rows[k] for k in sorted(self.rows)]) != self.selected_manifest_sha256 or
                digest(self.decompositions) != self.decomposition_sha256 or
                digest([self.seed,self.kind,self.length,self.batch_pairs,self.keys,self.systems,
                        self.episodes,self.episode_count,self.requested_frames,self.windows]) != self._settings):
            raise ValueError('A schedule metadata, settings or relation graph changed')

    def inventory(self):
        self._guard()
        return dict(schema=SCHEMA, status='CANDIDATE_NOT_FROZEN', seed=self.seed, kind=self.kind,
            history_frames=self.length, strict_A_compatible=self.length == 24, pairs_per_batch=self.batch_pairs,
            systems=len(self.keys), episodes_per_system=self.episode_count, requested_frames=self.requested_frames,
            temporal_templates=len(self.windows), batches_per_sweep=len(self.keys)//self.batch_pairs,
            relation_candidates={r: len(v[0]) for r, v in self.candidates.items()},
            random_candidates=len(self.keys)-1, decomposition_sha256=self.decomposition_sha256,
            manifest_sha256=self.manifest_sha256, selected_manifest_sha256=self.selected_manifest_sha256,
            template_policy='uniform over legal ordered disjoint supports, complete h16 targets for both branches',
            donor_policy='exactly balanced systems per sweep; uniform admissible edge marginals, dependent pairs within sweep',
            episode_policy='shared recipient permutation and nonzero cyclic offset over each complete episode cycle',
            shortfall_policy='retain planned identities and stop; no rejection resampling or successful-only population',
            test_read=False, formal_training=False)

    def sweep(self, sweep, name):
        self._guard()
        if type(sweep) is not int or not 0 <= sweep < 2**32: raise ValueError('invalid registered sweep index')
        config = configuration(name); relation = config['relation']; n = len(self.keys)
        episode_cycle, position = divmod(sweep, self.episode_count)
        erng = self._rng(episode_cycle, 1); order = erng.permutation(self.episode_count)
        offset = int(erng.integers(1, self.episode_count))
        recipient_rep = int(order[position]); independent_rep = int(order[(position+offset)%self.episode_count])
        donor_start, recipient_start = self.windows[int(self._rng(sweep, 2).integers(len(self.windows)))]
        if relation in self.decompositions:
            color = int(self._rng(sweep, 3 if relation == 'G1' else 4).integers(len(self.decompositions[relation])))
            mapping = self.decompositions[relation][color]; candidate_count = len(self.decompositions[relation])
        elif relation == 'Random':
            # Each nonzero cyclic shift hits every donor exactly once; uniform
            # shifts give every incorrect system equal conditional probability.
            color = int(self._rng(sweep, 5).integers(1, n)); mapping = [(i+color)%n for i in range(n)]; candidate_count = n-1
        else: color = 0; mapping = list(range(n)); candidate_count = 1
        pairs = []
        for i in self._rng(sweep, 6).permutation(n).tolist():
            recipient_system = self.keys[i]; donor_system = self.keys[mapping[i]]
            donor_rep = recipient_rep if config['pairing_profile'] == 'SameEp' else independent_rep
            recipient = self.episodes[recipient_system][recipient_rep]; donor = self.episodes[donor_system][donor_rep]
            donor_window = WindowSupport(donor, donor_start, self.length)
            recipient_window = WindowSupport(recipient, recipient_start, self.length)
            support = validate_pair(donor_window, recipient_window, pairing_profile=config['pairing_profile'])
            a = dict(split='train', theta=dict(zip(('m','gamma','k'), self.systems[donor_system])))
            b = dict(split='train', theta=dict(zip(('m','gamma','k'), self.systems[recipient_system])))
            if relation != 'Random': validate_relation(a, b, relation)
            elif donor_system == recipient_system: raise ValueError('random wrong relation has an exact match')
            pair = dict(recipient_system=recipient_system, donor_system=donor_system,
                donor_episode=donor, recipient_episode=recipient, donor_start=donor_start, recipient_start=recipient_start,
                history_frames=self.length, horizons=config['horizons'], support=support,
                candidate_count=candidate_count, conditional_system_probability=1./candidate_count,
                recipient_episode_probability=1./self.episode_count,
                conditional_donor_episode_candidates=1 if config['pairing_profile'] == 'SameEp' else self.episode_count-1,
                conditional_donor_episode_probability=1. if config['pairing_profile'] == 'SameEp' else 1./(self.episode_count-1),
                shared_factors=[k for k in ('m','gamma','k') if a['theta'][k] == b['theta'][k]],
                changed_factors=[k for k in ('m','gamma','k') if a['theta'][k] != b['theta'][k]])
            pair['pair_sha256'] = digest(pair); pairs.append(pair)
        result = dict(schema=SCHEMA, configuration=config, sweep=sweep, sampling_seed=self.seed,
            manifest_sha256=self.manifest_sha256, decomposition_sha256=self.decomposition_sha256,
            relation_matching_index=color, recipient_replicate=recipient_rep, independent_donor_replicate=independent_rep,
            temporal_probability=1./len(self.windows), pairs_per_batch=self.batch_pairs, pairs=pairs)
        result['plan_sha256'] = digest(result)
        return result

    def validate(self, plan):
        if digest(plan) != digest(self.sweep(plan['sweep'], plan['configuration']['name'])):
            raise ValueError('A training plan differs from the registered common schedule')
        return True
