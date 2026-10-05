"""Deterministic focal-one-to-many train plans from audited relation metadata.

No observation/target arrays enter sampling. A key identifies shared physical
attribute classes, not verified identity of one object across experiments.
Donor attributes are inferred only from membership in manifest candidate pools;
an inactive CD object's zero-filled metadata are never used as its donor label.
"""
from collections import Counter, defaultdict
import hashlib

import numpy as np


VERSION = 'collision-focal-multiquery-sampler-v4.9-1'


def _key_record(key):
    return [int(key[0]), int(key[1]), [int(value) for value in key[2]]]


def _plan_hash(epoch, supports, queries_per_group, arrays):
    h = hashlib.sha256()
    h.update(f'{VERSION}:{epoch}:{supports}:{queries_per_group}'.encode())
    for name, value in arrays.items():
        value = np.ascontiguousarray(value)
        h.update(name.encode())
        h.update(str(value.shape).encode())
        h.update(str(value.dtype).encode())
        h.update(value.tobytes())
    return h.hexdigest()


class MultiQuerySampler:
    def __init__(self, data, supports=5, queries_per_group=3):
        if not isinstance(supports, int) or supports < 1:
            raise ValueError('supports must be a positive integer')
        if not isinstance(queries_per_group, int) or queries_per_group < 2:
            raise ValueError('queries_per_group must be an integer >=2')
        self.supports = supports
        self.queries_per_group = queries_per_group
        row = data.rows['train']
        part = data.manifest['splits']['train']
        self.ids = list(map(str, row['ids']))
        self.all_ids = list(map(str, part['all_ids']))
        if len(set(self.ids)) != len(self.ids) or len(set(self.all_ids)) != len(self.all_ids):
            raise ValueError('Duplicate episode IDs in query or donor inventory')
        if self.ids != list(map(str, part['query_ids'])):
            raise ValueError('Query row order differs from the manifest')
        self.mask = np.asarray(row['mask']) > 0
        if self.mask.ndim != 2 or self.mask.shape[0] != len(self.ids):
            raise ValueError('Query activity mask does not align with IDs')
        self.n, self.k = self.mask.shape
        id_index = {ident: i for i, ident in enumerate(self.all_ids)}
        self.query_source_indices = np.asarray([id_index[ident] for ident in self.ids], dtype=np.int64)
        physical = np.asarray(data.physical['train'])
        public = np.asarray(data.public['train'])
        if physical.shape != (self.n, self.k, 3) or public.shape != self.mask.shape:
            raise ValueError('Audited active-query metadata have unexpected shapes')
        self.query_keys = {}
        self.query_pool = defaultdict(list)
        self.source_pool = defaultdict(set)
        self.candidates = {}
        self.source_key = {}
        self.active_slots = []
        for i, ident in enumerate(self.ids):
            active = np.flatnonzero(self.mask[i]).tolist()
            if not active:
                raise ValueError('Train query has no active focal object: ' + ident)
            self.active_slots.append(active)
            for slot in active:
                known = np.asarray(part['known_type'][ident][slot])
                if known.ndim != 1 or len(known) == 0 or not np.isclose(known.sum(), 1):
                    raise ValueError('Active object lacks a known public type')
                kind = int(known.argmax())
                if int(public[i, slot]) != slot * len(known) + kind:
                    raise ValueError('Public type/slot strata differ from the manifest')
                properties = tuple(int(value) for value in physical[i, slot])
                if properties != tuple(part['physical'][ident][slot]):
                    raise ValueError('Active query properties differ from the manifest')
                key = (slot, kind, properties)
                self.query_keys[i, slot] = key
                self.query_pool[key].append(i)
                candidate = set(map(int, part['candidates'][ident][slot]))
                if not candidate or min(candidate) < 0 or max(candidate) >= len(self.all_ids):
                    raise ValueError('Invalid donor candidate index')
                if int(self.query_source_indices[i]) in candidate:
                    raise ValueError('Manifest already contains a self-episode donor')
                self.candidates[i, slot] = candidate
                self.source_pool[key].update(candidate)
                for donor in candidate:
                    old_key = self.source_key.setdefault((donor, slot), key)
                    if old_key != key:
                        raise ValueError('One donor slot has inconsistent keys in candidate-pool memberships')
        self.keys = sorted(self.source_pool)
        self.key_indices = {key: i for i, key in enumerate(self.keys)}
        self.wrong_keys = {key: [other for other in self.keys
                                if other[:2] == key[:2] and other[2] != key[2]]
                           for key in self.keys}
        for key, pool in self.query_pool.items():
            if len(pool) < queries_per_group:
                raise ValueError(f'Fewer than {queries_per_group} distinct queries for active key {key}: {len(pool)}')
            if not self.wrong_keys[key]:
                raise ValueError('No different-physical-key donor pool for public stratum: ' + str(key))

    def make_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 1:
            raise ValueError('epoch must be a positive integer')
        seed = int.from_bytes(hashlib.sha256(f'{VERSION}:epoch:{epoch}'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        assigned = defaultdict(list)
        for i, slots in enumerate(self.active_slots):
            slot = int(rng.choice(slots))
            assigned[self.query_keys[i, slot]].append(i)
        groups = []
        padded = []
        primary_counts = {}
        for key in sorted(assigned):
            values = rng.permutation(np.asarray(assigned[key], dtype=np.int64)).tolist()
            primary_counts[key] = len(values)
            for start in range(0, len(values), self.queries_per_group):
                group = values[start:start + self.queries_per_group]
                missing = self.queries_per_group - len(group)
                if missing:
                    choices = [i for i in self.query_pool[key] if i not in group]
                    if len(choices) < missing:
                        raise ValueError('Cannot pad group with distinct same-key queries')
                    extra = rng.choice(choices, missing, replace=False).tolist()
                    group.extend(extra)
                    padded.extend(extra)
                groups.append((key, group))
        order = rng.permutation(len(groups))
        groups = [groups[int(i)] for i in order]
        g = len(groups)
        q = g * self.queries_per_group
        query_indices = np.empty(q, dtype=np.int64)
        focal = np.empty(q, dtype=np.int64)
        independent = np.zeros((q, self.k, 2, self.supports), dtype=np.int64)
        shared_correct = np.empty((g, self.supports), dtype=np.int64)
        shared_wrong = np.empty((g, self.supports), dtype=np.int64)
        group_keys, wrong_group_keys = [], []
        wrong_counts = Counter()
        for group_index, (key, indices) in enumerate(groups):
            if len(set(indices)) != self.queries_per_group:
                raise ValueError('Repeated query inside a shared-memory group')
            slot = key[0]
            excluded = set(int(self.query_source_indices[i]) for i in indices)
            if any(self.query_keys.get((i, slot)) != key for i in indices):
                raise ValueError('Shared focal queries do not have the same audited key')
            # Respect each recipient's allowed candidates, not just a generic key union.
            shared_pool = set.intersection(*(self.candidates[i, slot] for i in indices)) - excluded
            if len(shared_pool) < self.supports:
                raise ValueError(f'Insufficient shared-correct pool for group {group_index}, key {key}')
            shared = rng.choice(sorted(shared_pool), self.supports, replace=False)
            shared_correct[group_index] = shared
            reserved = set(map(int, shared))
            eligible_wrong = []
            for wrong_key in self.wrong_keys[key]:
                pool = self.source_pool[wrong_key] - excluded
                if len(pool) >= self.supports:
                    eligible_wrong.append((wrong_key, pool))
            if not eligible_wrong:
                raise ValueError(f'No eligible wrong-key pool for group {group_index}, key {key}')
            wrong_key, wrong_pool = eligible_wrong[int(rng.integers(len(eligible_wrong)))]
            wrong = rng.choice(sorted(wrong_pool), self.supports, replace=False)
            shared_wrong[group_index] = wrong
            if wrong_key[:2] != key[:2] or wrong_key[2] == key[2]:
                raise ValueError('Wrong key violates public matching or is physically correct')
            if any(self.source_key[int(donor), slot] != wrong_key for donor in wrong):
                raise ValueError('Wrong donor membership does not agree with inferred key')
            wrong_counts[wrong_key] += 1
            group_keys.append(self.key_indices[key])
            wrong_group_keys.append(self.key_indices[wrong_key])
            for j, i in enumerate(indices):
                dest = group_index * self.queries_per_group + j
                query_indices[dest] = i
                focal[dest] = slot
                for object_slot in self.active_slots[i]:
                    pool = self.candidates[i, object_slot] - excluded
                    # Reserve shared first, so it cannot exhaust the union of independently drawn sets.
                    if object_slot == slot:
                        pool = pool - reserved
                    if len(pool) < 2 * self.supports:
                        raise ValueError(f'Insufficient independent support after exclusions: group={group_index}, '
                                         f'query={self.ids[i]}, slot={object_slot}, available={len(pool)}, '
                                         f'required={2 * self.supports}')
                    draw = rng.choice(sorted(pool), 2 * self.supports, replace=False).reshape(2, self.supports)
                    independent[dest, object_slot] = draw
                    if excluded.intersection(map(int, draw.reshape(-1))):
                        raise ValueError('One of the group queries entered an independent support set')
                if reserved.intersection(map(int, independent[dest, slot].reshape(-1))):
                    raise ValueError('Shared focal memory overlaps an independent focal support set')
            if excluded.intersection(map(int, shared)) or excluded.intersection(map(int, wrong)):
                raise ValueError('Shared history contains a group query episode')
        counts = np.bincount(query_indices, minlength=self.n)
        if len(counts) != self.n or np.any(counts < 1):
            raise ValueError('Epoch plan did not include every original train query')
        arrays = {'query_indices': query_indices, 'focal': focal, 'independent': independent,
                  'shared_correct': shared_correct, 'shared_wrong': shared_wrong}
        summary = {'version': VERSION, 'epoch': epoch, 'seed': seed,
            'original_queries': self.n, 'query_occurrences': q, 'groups': g,
            'queries_per_group': self.queries_per_group, 'supports': self.supports,
            'padding_occurrences': len(padded), 'extra_query_occurrences': q - self.n,
            'repeated_distinct_queries': int(np.sum(counts > 1)),
            'maximum_query_occurrences': int(counts.max()),
            'padding_query_indices': list(map(int, padded)),
            'every_original_query_covered': True, 'same_key_does_not_imply_same_object_identity': True,
            'all_group_queries_excluded_from_all_group_supports': True,
            'independent_sets_disjoint_per_query_object': True,
            'shared_correct_disjoint_from_both_independent_focal_sets': True,
            'wrong_is_one_different_physical_key_with_same_slot_and_type': True,
            'wrong_group_coverage': g, 'wrong_accidentally_correct_groups': 0,
            'metadata_only_sampling': True, 'donor_metadata_source': 'union of audited candidate memberships',
            'keys': [_key_record(key) for key in self.keys],
            'source_pool_sizes': [len(self.source_pool[key]) for key in self.keys],
            'query_pool_sizes': [len(self.query_pool[key]) for key in self.keys],
            'primary_focal_assignments_by_key': [primary_counts.get(key, 0) for key in self.keys],
            'correct_key_indices_by_group': group_keys, 'wrong_key_indices_by_group': wrong_group_keys,
            'wrong_key_group_counts': [wrong_counts.get(key, 0) for key in self.keys]}
        return {**arrays, 'summary': summary,
                'plan_sha256': _plan_hash(epoch, self.supports, self.queries_per_group, arrays)}
