"""Shared real-field interpretation for train/val audit and released test prep.

Only AB and initial C enter eligibility. No future trajectory or method output
is used to decide metadata classes, object identity or relation coverage.
"""
import hashlib
import re
from pathlib import Path
import numpy as np
from cophy_metadata import blocktower_confounds_by_color, parse_gravity
from cophy_protocol import category_index

PLANS={
    'collision':dict(folder='collisionCF',count=None,variant='normal',slots=4,frames=15,
        colors=['yellow','green','blue','red'],fields=['mass','friction','restitution'],
        support=[[1,2,5],[.1,.5,1],[.1,.5,1]]),
    'balls':dict(folder='ballsCF',count=4,variant=None,slots=9,frames=30,
        colors=['red','green','blue','yellow','lime','purple','orange','cyan','magenta'],
        fields=['mass','friction','restitution'],support=[[1,2,5],[.1,.5,1],[.1,.5,1]]),
    'blocktower':dict(folder='blocktowerCF',count=3,variant='normal',slots=4,frames=30,
        colors=['red','green','blue','yellow'],fields=['mass','friction'],support=[[1,10],[.5,1]])}


def inspect_episode(data_root,scene,split,ident):
    if scene not in PLANS or split not in {'train','val','test'}:raise ValueError('Unknown scene/split')
    plan=PLANS[scene];path=Path(data_root)/plan['folder']
    if plan['count'] is not None:path=path/str(plan['count'])
    path=path/str(ident)
    conf=np.load(path/'confounders.npy',allow_pickle=False)
    ab=np.load(path/'ab/states.npy',allow_pickle=False,mmap_mode='r')
    c=np.load(path/'cd/states.npy',allow_pickle=False,mmap_mode='r')[0]
    if conf.shape!=(plan['slots'],len(plan['fields'])) or ab.ndim!=3 or ab.shape[:2]!=(plan['frames'],plan['slots']):
        raise ValueError(f'Invalid field shapes: conf={conf.shape}, AB={ab.shape}')
    if c.shape[0]!=plan['slots'] or c.shape[1]<3 or ab.shape[-1]<3:raise ValueError('C/AB slot or position mismatch')
    a0=np.asarray(ab[0,:,:3]);c0=np.asarray(c[:,:3])
    if not all(np.isfinite(x).all() for x in [conf,a0,c0]):raise ValueError('Nonfinite initial observations or metadata')
    pa=np.abs(a0).sum(-1)>0;pc=np.abs(c0).sum(-1)>0
    gravity=(path/'gravity.txt').read_text().strip() if scene=='blocktower' else None
    if scene=='blocktower':
        conf=blocktower_confounds_by_color(conf,(path/'ab/colors.txt').read_text(),pa)
        gravity_values=parse_gravity(gravity)
    types=['ball' if scene=='balls' else 'block']*plan['slots']
    if scene=='collision':
        types=['absent']*plan['slots']
        for line in (path/'ab/colors.txt').read_text().splitlines():
            obj=re.search(r'type=(\S+)',line);col=re.search(r'color=(\S+)',line)
            if obj and col:types[plan['colors'].index(col.group(1))]=obj.group(1)
        if any(types[k] not in {'sphere','cylinder_up','cylinder_down'} for k in np.flatnonzero(pa)):
            raise ValueError('Missing/unknown active Collision object type')
    rows=[]
    for slot in np.flatnonzero(pa):
        labels=[category_index(conf[slot,j],support) for j,support in enumerate(plan['support'])]
        row={'id':str(ident),'split':split,'slot':int(slot),'known_type':types[slot],
             'physical':labels,'raw_physical':conf[slot].astype(float).tolist(),'in_C':bool(pc[slot])}
        if scene=='blocktower':row['raw_gravity']=gravity_values
        rows.append(row)
    signature=hashlib.sha256(a0.astype('<f4').tobytes()+c0.astype('<f4').tobytes()).hexdigest()
    return {'id':str(ident),'split':split,'rows':rows,'gravity':gravity,
            'initial_observation_signature':signature,'AB_shape':list(ab.shape),'C_slots':int(pc.sum())}
