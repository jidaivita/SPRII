"""Content binding for development/training banks, including private labels.

This is a provenance check at explicit boundaries, not filesystem isolation.
Sealed test trajectories are forbidden here; their eventual evaluator needs a
separate entry point and permission contract.
"""
import argparse,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path,PurePosixPath

SCHEMA='vec.bank-content-snapshot.v1.1'
METADATA=('BANK_CONFIG.json','BANK_REPORT.json','MANIFEST.private.json','TRAIN_TARGET_STATISTICS.json')


def checked_asset(root,relative):
    root=Path(root).resolve();part=PurePosixPath(relative)
    if not relative or part.is_absolute() or '..' in part.parts or str(part)!=relative:
        raise ValueError('noncanonical asset path')
    path=root/relative
    if not path.resolve().is_relative_to(root):raise ValueError('asset escapes bank')
    current=root
    for component in part.parts:
        current=current/component
        if current.is_symlink():raise ValueError('symlink in bound bank asset')
    if not path.is_file():raise ValueError('bound asset missing: '+relative)
    return path


def stable_digest(path):
    """Reject replacement or writes observed while reading a file."""
    path=Path(path);before=path.stat();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    after=path.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if identity(before)!=identity(after):raise ValueError('asset changed during hashing: '+str(path))
    return dict(bytes=after.st_size,sha256=digest.hexdigest())


def declared_paths(root):
    root=Path(root);config=json.loads(checked_asset(root,'BANK_CONFIG.json').read_text())
    report=json.loads(checked_asset(root,'BANK_REPORT.json').read_text())
    manifest=json.loads(checked_asset(root,'MANIFEST.private.json').read_text())
    rows=manifest['episodes']
    if config.get('test_generated') or config.get('test_read') or report.get('test_read') or report.get('test_generated'):
        raise ValueError('test bank forbidden in training snapshot')
    if any(row['split'] not in ('train','validation') for row in rows):raise ValueError('test episode forbidden in training snapshot')
    if report['execution_errors'] or len(rows)!=config['job_count'] or len(rows)!=report['episodes']:
        raise ValueError('incomplete bank cannot be snapshotted')
    paths={name:'metadata' for name in METADATA}
    for row in rows:
        declared=[(row['private_state_path'],'private_state'),
                  (str(PurePosixPath(row['private_state_path']).with_name('private.json')),'episode_metadata')]
        declared.extend((asset['path'],'public_observation') for asset in row['assets'].values())
        for name,role in declared:
            checked_asset(root,name)
            if name in paths:raise ValueError('asset shared by distinct declared slots: '+name)
            paths[name]=role
    return paths,len(rows)


def content_digest(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def capture(root,workers=16):
    root=Path(root);began=time.monotonic()
    before={name:stable_digest(checked_asset(root,name)) for name in METADATA}
    paths,episodes=declared_paths(root)
    def inspect(item):
        name,role=item
        return dict(path=name,role=role,**stable_digest(checked_asset(root,name)))
    with ThreadPoolExecutor(max_workers=workers) as pool:files=list(pool.map(inspect,sorted(paths.items())))
    after={name:stable_digest(checked_asset(root,name)) for name in METADATA}
    if before!=after:raise ValueError('bank metadata changed during snapshot')
    return dict(schema=SCHEMA,status='PASS',files=files,content_sha256=content_digest(files),episodes=episodes,
        bytes=sum(f['bytes'] for f in files),seconds=time.monotonic()-began,test_read=False,
        scope='all manifest-declared pixels, actions, timestamps, private state targets and episode metadata; four root metadata files; excludes progress logs and this sidecar')


def verify(root,snapshot,workers=16):
    if snapshot.get('schema')!=SCHEMA or snapshot.get('status')!='PASS' or snapshot.get('test_read'):
        raise ValueError('invalid bank content snapshot')
    files=snapshot['files']
    if content_digest(files)!=snapshot.get('content_sha256'):raise ValueError('snapshot content digest mismatch')
    current=capture(root,workers)
    if current['files']!=files or current['episodes']!=snapshot.get('episodes'):
        raise ValueError('bank content differs from registered snapshot')
    return dict(status='PASS',content_sha256=current['content_sha256'],files=len(files),bytes=current['bytes'],
        seconds=current['seconds'],test_read=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('capture','verify'))
    parser.add_argument('--bank',type=Path,required=True);parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16);a=parser.parse_args()
    if a.mode=='capture':
        if a.snapshot.exists():raise ValueError('snapshot already exists')
        result=capture(a.bank,a.workers);a.snapshot.write_text(json.dumps(result,indent=2)+'\n')
    else:result=verify(a.bank,json.loads(a.snapshot.read_text()),a.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','scope')}),flush=True)


if __name__=='__main__':main()
