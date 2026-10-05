"""Fail-closed source, data and optimization bindings for formal Z training."""
import hashlib,json
from pathlib import Path


def file_digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint():
    digest=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()


def verify_admission(admission,bank):
    if admission.get('status')!='PASS' or admission.get('test_read') or admission.get('problems'):
        raise ValueError('data admission did not pass')
    for key,actual in (('manifest_sha256',bank.manifest_sha256),('target_statistics_sha256',file_digest(bank.root/'TRAIN_TARGET_STATISTICS.json')),
                       ('bank_config_sha256',file_digest(bank.root/'BANK_CONFIG.json'))):
        if admission.get(key)!=actual:raise ValueError('data admission binding differs: '+key)


def verify_formal(protocol,args,bank,admission,source=None):
    verify_admission(admission,bank)
    if protocol.get('status')!='FROZEN':raise ValueError('formal protocol is not frozen')
    if protocol.get('execution_entry')!='frozen_training' or getattr(args,'formal_entry',None)!='frozen_training':
        raise ValueError('formal training requires the before/after content-verifying entry')
    from .dataset_snapshot import verify
    snapshot_path=bank.root/'BANK_SNAPSHOT.json'
    if not snapshot_path.is_file():raise ValueError('formal training requires private/public content snapshot')
    snapshot=json.loads(snapshot_path.read_text())
    expected_bindings=dict(bank_manifest_sha256=bank.manifest_sha256,target_statistics_sha256=file_digest(bank.root/'TRAIN_TARGET_STATISTICS.json'),
        bank_config_sha256=file_digest(bank.root/'BANK_CONFIG.json'),source_fingerprint=source or source_fingerprint(),
        bank_snapshot_sha256=file_digest(snapshot_path),bank_content_sha256=snapshot.get('content_sha256'))
    for key,actual in expected_bindings.items():
        if protocol.get(key)!=actual:raise ValueError('formal artifact binding differs: '+key)
    if args.family not in protocol.get('families',[]):raise ValueError('unregistered model family')
    optimization=protocol.get('optimization',{})
    settings=dict(updates=args.max_updates,batch_size=args.batch_size,accumulation=args.accumulation,learning_rate=args.learning_rate,
        validate_every=args.validate_every,validation_batches=args.validation_batches,weight_decay=1e-4,gradient_clip=5.,warmup_updates=100,
        schedule='cosine_floor_0.1')
    for key,actual in settings.items():
        if optimization.get(key)!=actual:raise ValueError('formal optimization differs: '+key)
    if args.seed not in optimization.get('seeds',[]):raise ValueError('unregistered training seed')
    if protocol.get('resolution')!=args.resolution:raise ValueError('formal observation resolution differs')
    if args.validation_batches%80:raise ValueError('formal validation must cover complete paired query/budget/horizon cycles')
    if protocol.get('selection')!='paired_matched_null_all_query_budgets_all_horizons':raise ValueError('formal checkpoint selection differs')
    verify(bank.root,snapshot)
    return expected_bindings


def validation_case(index):
    """80 batches cover 2 cold/8 moving budgets ×4 horizons ×2 conditions.

    Adjacent matched/null batches use identical sampling seeds and complete
    cases. Only the context mask changes. Additional cycles draw new cases.
    """
    pairs=[(kind,budget,h) for kind,budgets in (('cold',(0,1)),('moving',(0,1,3,7,15,31,63,95))) for budget in budgets for h in (1,4,16,32)]
    pair=index//2;kind,budget,horizon=pairs[pair%len(pairs)]
    return dict(seed=988000000+pair,forced_context='forced96',forced_query=kind,
        forced_budget=budget,forced_horizon=horizon,forced_null=bool(index%2))
