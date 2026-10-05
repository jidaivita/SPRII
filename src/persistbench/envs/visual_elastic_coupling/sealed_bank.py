"""Evaluator-private test bank; deliberately separate from TrainingBank.

Opening requires frozen protocol/model commitments and an independently bound
bank admission. All declared public/private asset bytes are checked before and
after use. Model runners receive copied public packets, never this bank object.
"""
import collections,json,time
from functools import lru_cache
from pathlib import Path,PurePosixPath
import numpy as np
from .dataset_snapshot import checked_asset,stable_digest,content_digest
from .sealed_access import SealedAuthorization
from .planned_support import planned_jobs

SCHEMA='vec.sealed-bank-admission.v1'
METADATA=('BANK_CONFIG.json','BANK_REPORT.json','MANIFEST.private.json','TRAIN_TARGET_STATISTICS.json','PLANS.private.json')


def capture_content(root,authorization):
    """Called by the authorized generation/admission workflow, not training."""
    if not isinstance(authorization,SealedAuthorization):raise PermissionError('frozen authorization required for sealed content capture')
    root=Path(root);before={name:stable_digest(checked_asset(root,name)) for name in METADATA}
    manifest=json.loads((root/'MANIFEST.private.json').read_text());config=json.loads((root/'BANK_CONFIG.json').read_text());report=json.loads((root/'BANK_REPORT.json').read_text())
    rows=manifest['episodes']
    if config.get('schema')!='vec.sealed-bank.v1' or config.get('test_generated') is not True or any(r['split']!='test' for r in rows):raise ValueError('not a separately generated sealed bank')
    if report.get('execution_errors')!=[] or report.get('episodes')!=len(rows) or config.get('job_count')!=len(rows):raise ValueError('incomplete sealed generation')
    paths=set(METADATA)
    for row in rows:
        names=[row['private_state_path'],str(PurePosixPath(row['private_state_path']).with_name('private.json'))]+[a['path'] for a in row['assets'].values()]
        for name in names:
            if name in paths:raise ValueError('sealed asset reused by different declared slots')
            paths.add(name)
    files=[dict(path=name,**stable_digest(checked_asset(root,name))) for name in sorted(paths)]
    if before!={name:stable_digest(checked_asset(root,name)) for name in METADATA}:raise ValueError('sealed metadata changed during capture')
    return dict(files=files,content_sha256=content_digest(files),episodes=len(rows),bytes=sum(f['bytes'] for f in files))


class SealedBank:
    def __init__(self,root,authorization,*,admission_path,admission_sha256,audit_path,allow_labels=False,resolution=128):
        if not isinstance(authorization,SealedAuthorization):raise PermissionError('frozen authorization required before bank access')
        authorization.revalidate();self.authorization=authorization;self.root=Path(root);self.resolution=resolution
        self.admission_path=Path(admission_path);self.admission_sha256=admission_sha256
        if stable_digest(self.admission_path)['sha256']!=admission_sha256:raise ValueError('sealed admission commitment differs')
        admission=json.loads(self.admission_path.read_text())
        if admission.get('schema')!=SCHEMA or admission.get('status')!='PASS':raise PermissionError('sealed bank not admitted')
        if admission.get('protocol_sha256')!=authorization.protocol_sha256 or admission.get('selection_sha256')!=authorization.selection_sha256:raise ValueError('sealed data and model/protocol selection differ')
        current=capture_content(self.root,authorization)
        if current!=admission.get('content'):raise ValueError('sealed data content differs from admitted bytes')
        self.admission=admission;self._files={f['path']:f for f in current['files']};self.allow_labels=bool(allow_labels);self.closed=False
        self.config=self._json('BANK_CONFIG.json');self.report=self._json('BANK_REPORT.json');manifest=self._json('MANIFEST.private.json')
        if self.config.get('protocol_sha256')!=authorization.protocol_sha256 or self.config.get('selection_sha256')!=authorization.selection_sha256:raise ValueError('bank generation commitment differs')
        if self.config.get('source_fingerprint')!=authorization.protocol['source_fingerprint']:raise ValueError('bank generation source differs')
        if self.config.get('physics_config')!=authorization.protocol['physics_config'] or self.config.get('render_runtime')!=authorization.protocol['render_runtime']:
            raise ValueError('sealed physical/runtime profile differs')
        statistics_path=self._path('TRAIN_TARGET_STATISTICS.json')
        if stable_digest(statistics_path)['sha256']!=authorization.protocol['training_bank']['target_statistics_sha256']:raise ValueError('test bank does not use original training-only target statistics')
        self.statistics=json.loads(statistics_path.read_text())
        if self.statistics.get('source_split')!='train':raise ValueError('test normalization is forbidden')
        self.plans=self._json('PLANS.private.json');self.rows={};self.systems={};self.by_system=collections.defaultdict(lambda:collections.defaultdict(list))
        expected_jobs={}
        for name,entry in self.plans.items():
            namespace=authorization.validate_population(name,entry['plan'],entry['systems'])
            for job in planned_jobs(entry['plan'],entry['systems'],namespace):
                if job['episode_key'] in expected_jobs:raise ValueError('episode collision across sealed strata')
                expected_jobs[job['episode_key']]=job
        if set(self.plans)!=set(authorization.protocol['populations']):raise ValueError('sealed population matrix incomplete')
        for row in manifest['episodes']:
            if row['episode_key'] in self.rows:raise ValueError('duplicate sealed episode')
            expected=expected_jobs.get(row['episode_key'])
            if expected is None or any(row.get(k)!=v for k,v in expected.items()):raise ValueError('generated episode differs from frozen planned slot')
            if row['system_key']!=expected['system']['system_key'] or row['stratum']!=expected['system']['stratum'] or row['theta']!=expected['system']['theta']:
                raise ValueError('inconsistent sealed system metadata')
            self.rows[row['episode_key']]=row;key=('test',row['system_key']);self.systems[key]=row['system'];self.by_system[key][row['kind']].append(row)
        if set(self.rows)!=set(expected_jobs):raise ValueError('sealed planned episodes incomplete')
        self.keys={'test':sorted(self.by_system)};self.manifest_sha256=self._files['MANIFEST.private.json']['sha256']
        self.audit_path=Path(audit_path);self.audit_path.parent.mkdir(parents=True,exist_ok=True)
        self._audit_stream=self.audit_path.open('x');self._audit_sequence=0
        self._audit('OPEN_BANK',allow_labels=self.allow_labels,episodes=len(self.rows),admission_sha256=admission_sha256)

    def _audit(self,event,**values):
        self._audit_sequence+=1
        self._audit_stream.write(json.dumps(dict(sequence=self._audit_sequence,event=event,time_ns=time.time_ns(),**values))+'\n');self._audit_stream.flush()

    def _path(self,relative):
        if self.closed:raise PermissionError('sealed bank is closed')
        if relative not in self._files:raise ValueError('asset not included in sealed admission')
        path=checked_asset(self.root,relative);actual=stable_digest(path)
        if any(actual[k]!=self._files[relative][k] for k in ('bytes','sha256')):raise ValueError('sealed asset changed during evaluation')
        return path

    def _json(self,relative):return json.loads(self._path(relative).read_text())

    @lru_cache(maxsize=192)
    def _visible(self,key):
        row=self.rows[key]
        with np.load(self._path(row['assets'][str(self.resolution)]['path']),allow_pickle=False) as data:images=data['images'];actions=data['actions']
        if images.dtype!=np.uint8 or images.ndim!=3 or images.shape[1:]!=(self.resolution,self.resolution) or actions.shape!=(len(images)-1,2) or not np.isfinite(actions).all():
            raise ValueError('sealed public episode codec differs')
        images.setflags(write=False);actions.setflags(write=False);self._audit('READ_PUBLIC',episode_key=key,resolution=self.resolution)
        return images,actions

    def visible(self,key):
        if self.closed:raise PermissionError('sealed bank is closed')
        return self._visible(key)

    @lru_cache(maxsize=256)
    def _labels(self,key):
        with np.load(self._path(self.rows[key]['private_state_path']),allow_pickle=False) as data:state=data['state']
        state.setflags(write=False);self._audit('READ_LABEL',episode_key=key);return state

    def labels(self,key):
        if self.closed or not self.allow_labels:raise PermissionError('label-free extraction cannot read sealed targets')
        return self._labels(key)

    def verify_content(self):
        if self.closed:raise PermissionError('sealed bank is closed')
        self.authorization.revalidate()
        if stable_digest(self.admission_path)['sha256']!=self.admission_sha256 or capture_content(self.root,self.authorization)!=self.admission['content']:
            raise ValueError('sealed inputs changed before completion')

    def close(self):
        if self.closed:return
        try:
            self.verify_content()
            self._audit('CLOSE_BANK',status='PASS')
        except Exception as exc:
            self._audit('CLOSE_BANK',status='FAIL',error=repr(exc));raise
        finally:
            self.closed=True;self._visible.cache_clear();self._labels.cache_clear();self._audit_stream.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
