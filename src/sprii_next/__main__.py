"""Development CLI. Sealed loaders are deliberately not imported here."""
import argparse
import json
from pathlib import Path
from .io import load_protocol,read,write,sha,digest,npz,plain,development_path


def main():
    p=argparse.ArgumentParser(description='SPRII next-stage development infrastructure; no sealed-test command')
    sub=p.add_subparsers(dest='command',required=True)
    def command(name,help):return sub.add_parser(name,help=help)
    def arg(c,name,**kw):c.add_argument('--'+name,**kw)
    c=command('assemble','Bind source descriptors into one immutable development protocol')
    arg(c,'sources',nargs='+',required=True);arg(c,'output',required=True)
    c=command('plan','Print exact jobs without opening any data')
    arg(c,'environment',choices=['springworld','pokeworld'],required=True)
    arg(c,'stage',choices=['development','baseline','pilot','formal'],required=True);arg(c,'secondary',action='store_true');arg(c,'output')
    for name in ('bind-spring','export-spring'):
        c=command(name,'Verify a native frozen SpringWorld source and bind/export development features')
        for name2 in ('native-root','completion','method','output'):arg(c,name2,required=True)
        arg(c,'seed',type=int,required=True)
        if name=='bind-spring':
            for n in ('manifest','features','targets'):arg(c,n,required=True)
        else:arg(c,'bank',required=True);arg(c,'device',default='cuda:0')
    c=command('export-poke','Export G1/G2 frozen sources and fixed-action sensitivities')
    for n in ('assets','condition','output'):arg(c,n,required=True)
    arg(c,'seed',type=int,required=True);arg(c,'device',default='cuda:0');arg(c,'batch-size',type=int,default=64)
    c=command('preflight','Verify native assets, source identities, fixed endpoints and paired target vectors')
    arg(c,'protocol',required=True);arg(c,'environment',choices=['springworld','pokeworld'],required=True);arg(c,'output',required=True)
    arg(c,'stage',choices=['development','baseline','pilot','formal'])
    c=command('run','Run exactly one registered reader job')
    for n in ('protocol','root','environment','stage'):arg(c,n,required=True)
    arg(c,'job-index',type=int,required=True);arg(c,'device',default='cuda:0');arg(c,'gate');arg(c,'smoke-steps',type=int)
    arg(c,'secondary',action='store_true')
    c=command('geometry','Read source P only; save full donor vectors and light geometry')
    arg(c,'protocol',required=True);arg(c,'method',choices=['Structure','Align','Cross','Both'],required=True)
    arg(c,'seed',type=int,required=True);arg(c,'output',required=True)
    c=command('geometry-report','Aggregate the four-method by three-source frozen geometry grid')
    arg(c,'results',nargs='+',required=True);arg(c,'output',required=True)
    for name in ('spring-report','pilot-report','baseline-report'):
        c=command(name,'Aggregate complete paired grids by physical system')
        arg(c,'root',required=True);arg(c,'output',required=True)
        if name=='pilot-report':arg(c,'formal',action='store_true')
    c=command('gate','Record a reviewed development go/stop decision; never a p-value gate')
    arg(c,'summary',required=True);arg(c,'decision',choices=['go','stop'],required=True)
    arg(c,'reason',required=True);arg(c,'nonflat',action='store_true');arg(c,'output',required=True)
    c=command('effects','Gated frozen-reader sensitivity and optional reviewed target alignment')
    for n in ('protocol','method','run-dir','gate','output'):arg(c,n,required=True)
    arg(c,'seed',type=int,required=True);arg(c,'device',default='cpu');arg(c,'target-review');arg(c,'simulator-config')
    c=command('balls','Existing-vector reanalysis only; no new inference')
    arg(c,'vectors',required=True);arg(c,'output',required=True)
    a=p.parse_args();result=None
    if a.command=='assemble':
        from .protocol import default_protocol
        result=default_protocol([read(development_path(x)) for x in a.sources]);write(a.output,result);load_protocol(a.output)
    elif a.command=='plan':
        from .engine import jobs,job_name
        result=[dict(job_index=i,relative_output=job_name(j),**j) for i,j in enumerate(jobs(a.environment,a.stage,a.secondary))]
        if a.output:write(a.output,result)
    elif a.command in ('bind-spring','export-spring'):
        from .protocol import bind_spring,export_spring
        if a.command=='bind-spring':result=bind_spring(a.native_root,a.completion,a.manifest,a.features,a.targets,a.method,a.seed,a.output)
        else:result=export_spring(a.native_root,a.completion,a.bank,a.method,a.seed,a.output,a.device)
    elif a.command=='export-poke':
        from .poke_export import export
        result=export(a.assets,a.condition,a.seed,a.output,device=a.device,batch_size=a.batch_size)
    elif a.command=='preflight':
        from .protocol import preflight
        result=preflight(load_protocol(a.protocol),a.environment,a.stage);write(a.output,result)
    elif a.command=='run':
        from .engine import jobs,job_name,run_job
        from .providers import provider
        cfg=load_protocol(a.protocol);grid=jobs(a.environment,a.stage,a.secondary)
        if not 0<=a.job_index<len(grid):raise ValueError('job index outside registered grid')
        job=grid[a.job_index]
        if a.stage=='formal':
            from .statistics import verify_gate
            if a.gate is None:raise PermissionError('formal expansion requires a reviewed pilot go')
            verify_gate(a.gate,digest(cfg))
        source=provider(cfg,a.environment,job['method'],job['source_seed'])
        out=Path(a.root)/('smoke' if a.smoke_steps else 'runs')/job_name(job)
        run_job(source,cfg,job,out,device=a.device,smoke_steps=a.smoke_steps,gate_path=a.gate)
        result=dict(output=str(out),test_read=False,smoke=a.smoke_steps is not None)
    elif a.command=='geometry':
        from .providers import provider
        from .geometry import source_geometry
        source=provider(load_protocol(a.protocol),'springworld',a.method,a.seed)
        result=source_geometry(source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
        p,t,s,d=source.donors('validation')
        npz(out/'source_vectors.npz',persistent_code=p,physical_parameters=t,system_id=s,donor_id=d)
        result['source_vectors_sha256']=sha(out/'source_vectors.npz');write(out/'RESULT.json',result)
    elif a.command=='geometry-report':
        from .geometry import geometry_summary
        result=geometry_summary(a.results);write(a.output,result)
    elif a.command in ('spring-report','pilot-report','baseline-report'):
        from .statistics import spring_summary,pilot_summary,baseline_summary
        if a.command=='spring-report':result=spring_summary(a.root)
        elif a.command=='baseline-report':result=baseline_summary(a.root)
        else:result=pilot_summary(a.root,a.formal)
        write(a.output,result)
    elif a.command=='gate':
        from .statistics import verify_gate
        summary=read(a.summary)
        if summary['stage']!='pilot':raise ValueError('pilot summary required')
        if a.decision=='go' and (not a.nonflat or max(summary['positive_slopes'],summary['negative_slopes'])<2):raise ValueError('go requires reviewed nonflat structure and 2/3 consistent slopes')
        result=dict(decision=a.decision,trend_review=a.reason,binned_curve_nonflat=a.nonflat,p_value_used=False,
            majority_direction='positive' if summary['positive_slopes']>=2 else ('negative' if summary['negative_slopes']>=2 else 'none'),
            summary=str(Path(a.summary).resolve()),summary_sha256=sha(a.summary),test_read=False)
        write(a.output,result)
        if a.decision=='go':verify_gate(a.output)
    elif a.command=='effects':
        from .providers import provider
        from .effects import effect_analysis
        source=provider(load_protocol(a.protocol),'pokeworld',a.method,a.seed)
        effect_analysis(source,a.run_dir,a.gate,a.output,device=a.device,target_alignment_review=a.target_review,
                        simulator_config=None if a.simulator_config is None else read(a.simulator_config))
        result=read(Path(a.output)/'RESULT.json')
    elif a.command=='balls':
        from .effects import balls_reanalysis
        result=balls_reanalysis(development_path(a.vectors));write(a.output,result)
    print(json.dumps(plain(result),indent=2,allow_nan=False))


if __name__=='__main__':main()
