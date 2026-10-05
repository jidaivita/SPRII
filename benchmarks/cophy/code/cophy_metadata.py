"""Published-file metadata semantics, kept outside learned-model inputs."""
import re
import numpy as np

BLOCK_COLORS = ['red', 'green', 'blue', 'yellow']


def blocktower_confounds_by_color(raw, color_text, ab_presence):
    """CoPhy_224 stores confounders by object ID, states by color slot.

    Use AB's object-number/color table, since a C intervention may remove a
    block. Never infer the permutation from future trajectories or model error.
    The published 3-block train examples retain zero padding after object IDs.
    """
    raw = np.asarray(raw)
    active = np.asarray(ab_presence) > 0
    if raw.shape != (4, 2) or active.shape != (4,) or not np.isfinite(raw).all():
        raise ValueError('Invalid Blocktower physical metadata shape')
    mapping = []
    for line in color_text.splitlines():
        match = re.search(r'obj=(\d+)\s+color=(\w+)', line)
        if match:
            obj, color = int(match.group(1)), match.group(2)
            if color not in BLOCK_COLORS:
                raise ValueError('Unknown Blocktower color')
            mapping.append((obj, BLOCK_COLORS.index(color)))
    n = len(mapping)
    if (not 1 <= n <= 4 or sorted(i for i, _ in mapping) != list(range(n)) or
            len({k for _, k in mapping}) != n or
            {k for _, k in mapping} != set(np.flatnonzero(active))):
        raise ValueError('AB object IDs/colors disagree with observed active slots')
    if not (raw[:n] > 0).all() or np.any(raw[n:] != 0):
        raise ValueError('Blocktower confounders do not follow audited object-ID rows and zero padding')
    aligned = np.zeros_like(raw)
    for obj, slot in mapping:
        aligned[slot] = raw[obj]
    return aligned


def parse_gravity(text):
    values = []
    for axis in ('x', 'y'):
        match = re.search(r'gravity_'+axis+r'=([^\s]+)', text)
        if match is None:
            raise ValueError('Missing named gravity component')
        value = float(match.group(1))
        if not np.isfinite(value):
            raise ValueError('Nonfinite gravity component')
        values.append(value)
    return values
